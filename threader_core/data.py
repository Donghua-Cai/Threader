from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import DatasetSchema


ASSISTANT_BOILERPLATE = [
    re.compile(r"^i(?:'| a)m happy to help", re.I),
    re.compile(r"^as an ai", re.I),
    re.compile(r"^i do not have personal opinions", re.I),
]

GREETINGS = {
    "hi",
    "hello",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "好的",
    "谢谢",
    "嗯",
    "嗯嗯",
    "哈哈",
}


def parse_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except Exception:
        return None


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_sample_files(
    root: str | Path,
    question_type: Optional[str] = None,
    sample_id: Optional[str] = None,
    question_dir: Optional[str | Path] = None,
    schema: Optional[DatasetSchema] = None,
) -> List[Path]:
    base = Path(root)
    filter_type = bool(question_type and question_type.upper() != "ALL")
    if filter_type and (base / question_type).is_dir():
        base = base / question_type
    files = sorted(p for p in base.rglob("*.json") if p.name != "_manifest.json")
    if sample_id:
        files = [p for p in files if sample_id_from_path(p) == sample_id]
    if filter_type:
        question_files = index_question_files(question_dir) if question_dir else {}
        files = [p for p in files if matches_question_type(p, base, question_type, question_files, schema)]
    return files


def matches_question_type(
    path: Path,
    root: Path,
    question_type: str,
    question_files: Optional[Dict[str, Path]] = None,
    schema: Optional[DatasetSchema] = None,
) -> bool:
    if root.name == question_type or question_type in path.relative_to(root).parts[:-1]:
        return True
    actual_type = extract_question_meta(load_json(path), schema=schema).get("question_type")
    if not actual_type and question_files:
        question_path = question_files.get(sample_id_from_path(path))
        if question_path:
            actual_type = extract_question_meta(load_json(question_path), schema=schema).get("question_type")
    if actual_type:
        return str(actual_type) == question_type
    if path.parent != root:
        return False
    raise ValueError(
        f"Cannot filter flat sample {path} by question type: add question_type "
        "to the sample JSON or supply --question-dir with question metadata. "
        "Use --question-type ALL to process all samples."
    )


def sample_id_from_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def load_sample_id_file(path: str | Path) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        sid = raw.strip()
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def index_question_files(question_dir: str | Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for path in sorted(Path(question_dir).rglob("*.json")):
        if path.name == "_manifest.json":
            continue
        sid = sample_id_from_path(path)
        idx.setdefault(sid, path)
    return idx


def extract_question_meta(obj: Dict[str, Any], sample_id: str = "", schema: Optional[DatasetSchema] = None) -> Dict[str, Any]:
    qa_obj = obj.get("qa") if isinstance(obj.get("qa"), dict) else {}
    meta = obj.get("meta") or qa_obj.get("meta") or {}
    meta = meta if isinstance(meta, dict) else {}
    mcq = meta.get("multiple_choice") if isinstance(meta.get("multiple_choice"), dict) else {}
    options = obj.get("answer_options")
    if not isinstance(options, list):
        options = mcq.get("all_options") or []
    result = {
        "sample_id": str(sample_id),
        "question_id": obj.get("question_id"),
        "dataset": obj.get("dataset") or meta.get("dataset"),
        "question": obj.get("question"),
        "question_type": obj.get("question_type"),
        "question_date": obj.get("question_date"),
        "answer": obj.get("answer"),
        "answer_options": list(options or []),
        "correct_answer_label": obj.get("correct_answer_label") or mcq.get("correct_answer_label"),
    }
    return schema.question_meta(obj, result) if schema is not None else result


def load_question_meta(question_dir: str | Path, sample_id: str, schema: Optional[DatasetSchema] = None) -> Dict[str, Any]:
    path = index_question_files(question_dir).get(str(sample_id))
    if not path:
        return {}
    return extract_question_meta(load_json(path), sample_id=sample_id, schema=schema)


def find_segment_sample_path(segment_dir: str | Path, sample_id: str, question_type: Optional[str] = None) -> Optional[Path]:
    base = Path(segment_dir)
    candidates: List[Path] = []
    if question_type and question_type.upper() != "ALL":
        candidates.append(base / question_type / f"{sample_id}.json")
    candidates.append(base / f"{sample_id}.json")
    candidates.extend(sorted(base.rglob(f"{sample_id}.json")))
    for path in candidates:
        if path.is_file():
            return path
    suffixed: List[Path] = []
    if question_type and question_type.upper() != "ALL":
        suffixed.extend(sorted((base / question_type).glob(f"{sample_id}_*.json")))
    suffixed.extend(sorted(base.glob(f"{sample_id}_*.json")))
    suffixed.extend(sorted(base.rglob(f"{sample_id}_*.json")))
    for path in suffixed:
        if path.is_file():
            return path
    return None


def load_segment_sample(segment_dir: str | Path, sample_id: str, question_type: Optional[str] = None, schema: Optional[DatasetSchema] = None) -> Optional[Dict[str, Any]]:
    path = find_segment_sample_path(segment_dir, sample_id, question_type)
    if path is None:
        return None
    obj = load_json(path)
    if schema is None:
        return obj
    try:
        return schema.segment_sample(obj)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def expand_session_map(session_ids: Sequence[Any], session_num_segments: Sequence[Any]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for i, (sid, n) in enumerate(zip(session_ids or [], session_num_segments or [])):
        for _ in range(int(n or 0)):
            out.append((i, str(sid)))
    return out


def _looks_boilerplate(text: str) -> bool:
    return any(p.search(text.strip()) for p in ASSISTANT_BOILERPLATE)


def _is_chitchat(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return normalized in GREETINGS


def _truncate_text(text: str, head_chars: int = 1800, tail_chars: int = 600) -> str:
    if len(text) <= head_chars + tail_chars:
        return text
    return text[:head_chars] + "\n...[truncated]...\n" + text[-tail_chars:]


def build_segment_compact(segment_messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    user_parts: List[str] = []
    assistant_parts: List[str] = []
    full_parts: List[str] = []
    orders: List[int] = []
    times: List[str] = []
    for msg in segment_messages:
        role = str(msg.get("role") or "").strip()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        order = parse_int(msg.get("order"))
        if order is not None:
            orders.append(order)
        time = str(msg.get("time") or "").strip()
        if time:
            times.append(time)
        full_parts.append(f"[{role}]: {content}")
        role_l = role.lower()
        if role_l == "user":
            if _is_chitchat(content):
                continue
            user_parts.append(content)
        elif role_l == "assistant":
            if not _looks_boilerplate(content):
                assistant_parts.append(content)
    text_full = "\n".join(full_parts).strip()
    text_user = "\n".join(user_parts).strip() or text_full
    text_assistant = _truncate_text("\n".join(assistant_parts).strip())
    return {
        "text_user": text_user,
        "text_assistant": text_assistant,
        "text_full": text_full,
        "start_order": min(orders) if orders else None,
        "end_order": max(orders) if orders else None,
        "start_time": times[0] if times else None,
        "end_time": times[-1] if times else None,
    }


def extract_pair_turns(segment_messages: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    pairs: List[Dict[str, Any]] = []
    pending_user: Optional[Dict[str, Any]] = None
    orphan_assistant = 0
    for msg in segment_messages:
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending_user = msg
        elif role == "assistant":
            if pending_user is None:
                orphan_assistant += 1
                continue
            text_user = str(pending_user.get("content") or "").strip()
            text_assistant = content
            turn_order = parse_int(msg.get("order")) or parse_int(pending_user.get("order"))
            turn_time = str(msg.get("time") or pending_user.get("time") or "").strip() or None
            pairs.append(
                {
                    "text_user": text_user,
                    "text_assistant": text_assistant,
                    "text_turn": f"[user]: {text_user}\n[assistant]: {text_assistant}",
                    "turn_order": turn_order,
                    "turn_time": turn_time,
                }
            )
            pending_user = None
    dangling_user = 1 if pending_user is not None else 0
    return pairs, dangling_user, orphan_assistant
