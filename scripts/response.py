#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threader_core.api import api_settings, build_client, load_runtime_config
from threader_core.config import load_yaml
from threader_core.data import extract_question_meta, load_json, load_question_meta, load_segment_sample
from threader_core.schema import DatasetSchema

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None, unit=None):
        return iterable if iterable is not None else range(total or 0)


@dataclass
class SegmentRec:
    segment_id: str
    start_time: Optional[str]
    start_order: Optional[int]
    text_full: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate answers from exported query_*.json files.")
    p.add_argument("--config", default=str(ROOT / "config" / "baseline.yaml"), help="Use the same dataset field mapping as build_index.py and query.py.")
    p.add_argument("--query-dir", required=True)
    p.add_argument("--query-json", default=None)
    p.add_argument("--sample-id", default="ALL")
    p.add_argument("--question-type", default="ALL")
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--question-dir", default=None, help="Optional QA directory. If omitted, QA fields are read from query JSON or segment samples.")
    p.add_argument("--segment-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--runtime-config", default=str(ROOT / "config" / "runtime_api.example.yaml"))
    p.add_argument("--response-model", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-mode", choices=["none", "llm", "code_mcq", "auto"], default="auto")
    p.add_argument("--response-prompt-file", default=str(ROOT / "config" / "response_prompt.txt"))
    p.add_argument("--judge-prompt-file", default=str(ROOT / "config" / "judge_prompt.txt"))
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def discover_query_files(args: argparse.Namespace) -> List[Path]:
    if args.query_json:
        return [Path(args.query_json)]
    root = Path(args.query_dir)
    if args.sample_id and args.sample_id != "ALL":
        hits = sorted(root.rglob(f"query_{args.sample_id}.json"))
        return hits[:1]
    files = sorted(root.rglob("query_*.json"))
    if args.question_type and args.question_type.upper() != "ALL":
        kept: List[Path] = []
        for path in files:
            try:
                obj = load_json(path)
            except Exception:
                continue
            if (
                str(obj.get("question_type") or "") == str(args.question_type)
                or root.name == args.question_type
                or args.question_type in path.relative_to(root).parts[:-1]
            ):
                kept.append(path)
        files = kept
    files = files[int(args.sample_offset or 0):]
    if args.sample_limit is not None:
        files = files[: int(args.sample_limit)]
    return files


def load_prompt(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if "{question}" not in text or "{history_session}" not in text:
        raise ValueError("response prompt must contain {question} and {history_session}")
    return text


def fetch_segments(db_path: str, segment_ids: List[str]) -> List[SegmentRec]:
    if not segment_ids:
        return []
    conn = sqlite3.connect(db_path)
    qmarks = ",".join(["?"] * len(segment_ids))
    rows = conn.execute(f"SELECT segment_id, start_time, start_order, text_full FROM segments WHERE segment_id IN ({qmarks})", segment_ids).fetchall()
    conn.close()
    recs = [SegmentRec(r[0], r[1], r[2], r[3] or "") for r in rows]
    recs.sort(key=lambda r: ((r.start_time or ""), r.start_order if r.start_order is not None else -1, r.segment_id))
    return recs


def render_history_from_raw(segment_dir: Path, sample_id: str, question_type: Optional[str], segment_ids: List[str], schema: Optional[DatasetSchema] = None) -> Optional[str]:
    obj = load_segment_sample(segment_dir, sample_id, question_type, schema=schema)
    if not obj:
        return None
    all_segments = obj.get("segments_messages") or []
    lines: List[str] = []
    for sid in segment_ids:
        try:
            idx = int(str(sid).split(":", 1)[1])
        except Exception:
            continue
        if idx < 0 or idx >= len(all_segments):
            continue
        for msg in all_segments[idx]:
            role = str(msg.get("role") or "unknown").strip()
            content = re.sub(r"\s+", " ", str(msg.get("content") or "")).strip()
            if not content:
                continue
            t = str(msg.get("time") or "").strip()
            prefix = f"[{t}] [{role}]" if t else f"[{role}]"
            lines.append(f"{prefix} {content}")
    return "\n".join(lines)


def render_history_from_db(db_path: str, segment_ids: List[str]) -> str:
    lines: List[str] = []
    for rec in fetch_segments(db_path, segment_ids):
        for raw in rec.text_full.splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if line:
                lines.append(line)
    return "\n".join(lines)


def question_with_date(question: str, question_date: Optional[str]) -> str:
    q = str(question or "").strip()
    qd = str(question_date or "").strip()
    return f"[Question Date: {qd}] {q}" if q and qd and not q.startswith("[Question Date:") else q


def format_options(options: List[str]) -> str:
    return "\n".join(str(x).strip() for x in options if str(x).strip())


def normalize_choice(text: Optional[str]) -> Optional[str]:
    s = str(text or "").strip()
    m = re.search(r"\(\s*([A-Da-d])\s*\)", s) or re.search(r"^\s*([A-Da-d])[\)\.\:\-]?\s*$", s)
    return m.group(1).upper() if m else None


def derive_choice(pred: str, options: List[str]) -> Optional[str]:
    direct = normalize_choice(pred)
    if direct:
        return direct
    p = re.sub(r"\s+", " ", str(pred or "").lower()).strip()
    for i, opt in enumerate(options):
        o = re.sub(r"\s+", " ", str(opt or "").lower()).strip()
        if o and (p == o or p in o or o in p):
            return chr(ord("A") + i)
    return None


def gold_choice(label: Optional[str], answer: Optional[str], options: List[str]) -> Optional[str]:
    direct = normalize_choice(label)
    if direct:
        return direct
    ans = re.sub(r"\s+", " ", str(answer or "").lower()).strip()
    for i, opt in enumerate(options):
        o = re.sub(r"\s+", " ", str(opt or "").lower()).strip()
        if ans and o and (ans == o or ans in o):
            return chr(ord("A") + i)
    return None


def call_response(client, model: str, prompt_template: str, question: str, history: str, options: List[str]) -> str:
    prompt = prompt_template.format(question=question, history_session=history, answer_options=format_options(options))
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return (resp.choices[0].message.content or "").strip()


def call_judge(client, model: str, judge_prompt: str, question: str, pred: str, answer: Optional[str]) -> Dict[str, Any]:
    user = f"Question:\n{question}\n\nGold Answer:\n{answer or ''}\n\nPrediction:\n{pred}\n"
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": judge_prompt}, {"role": "user", "content": user}],
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {"is_correct": False, "verdict": "error", "reason": "judge returned invalid json"}


def process_one(qpath: Path, args: argparse.Namespace, response_client, judge_client, response_model: str, judge_model: str, prompt_template: str, judge_prompt: Optional[str], history_dir: Path) -> Dict[str, Any]:
    qobj = load_json(qpath)
    sample_id = str(qobj.get("sample_id") or qpath.stem.replace("query_", ""))
    schema = getattr(args, "data_schema", None)
    qmeta = load_question_meta(Path(args.question_dir), sample_id, schema=schema) if args.question_dir else {}
    if not qmeta.get("question"):
        seg_obj_for_meta = load_segment_sample(Path(args.segment_dir), sample_id, qobj.get("question_type"), schema=schema)
        qmeta = extract_question_meta(seg_obj_for_meta or {}, sample_id=sample_id, schema=schema)
    question = qobj.get("question") or qmeta.get("question")
    question_date = qobj.get("question_date") or qmeta.get("question_date")
    question_type = qobj.get("question_type") or qmeta.get("question_type")
    answer = qobj.get("answer") if qobj.get("answer") is not None else qmeta.get("answer")
    options = qobj.get("answer_options") or qmeta.get("answer_options") or []
    correct_label = qobj.get("correct_answer_label") if qobj.get("correct_answer_label") is not None else qmeta.get("correct_answer_label")
    segment_ids = [str(x) for x in (qobj.get("segment_ids") or qobj.get("retrieved_segment_ids") or [])]
    db_path = str(qobj.get("db_path") or "")
    # Match the original response pipeline: order history by segment time,
    history_segment_ids = [rec.segment_id for rec in fetch_segments(db_path, segment_ids)]
    history = render_history_from_raw(Path(args.segment_dir), sample_id, question_type, history_segment_ids, schema=schema)
    if history is None:
        history = render_history_from_db(db_path, segment_ids)
    q_for_prompt = question_with_date(str(question or ""), question_date)
    t0 = time.perf_counter()
    pred_raw = call_response(response_client, response_model, prompt_template, q_for_prompt, history, options)
    response_time = float(time.perf_counter() - t0)
    mode = args.judge_mode
    if mode == "auto":
        mode = "code_mcq" if gold_choice(correct_label, answer, options) else "llm"
    if mode == "none":
        judge = {"is_correct": None, "verdict": "not_judged"}
    elif mode == "code_mcq":
        pred_c = derive_choice(pred_raw, options)
        gold_c = gold_choice(correct_label, answer, options)
        judge = {"is_correct": bool(pred_c and gold_c and pred_c == gold_c), "predicted_choice": pred_c, "gold_choice": gold_c, "verdict": "correct" if pred_c and gold_c and pred_c == gold_c else "incorrect"}
    else:
        if not judge_prompt:
            raise ValueError("judge prompt required for llm judge")
        judge = call_judge(judge_client, judge_model, judge_prompt, q_for_prompt, pred_raw, answer)
    hist_path = history_dir / f"history_{sample_id}.txt"
    hist_path.write_text(history, encoding="utf-8")
    return {
        "sample_id": sample_id,
        "question_id": qobj.get("question_id") or qmeta.get("question_id"),
        "dataset": qobj.get("dataset") or qmeta.get("dataset"),
        "query_json": str(qpath),
        "question": question,
        "question_date": question_date,
        "question_type": question_type,
        "gold_answer": answer,
        "answer_options": options,
        "correct_answer_label": correct_label,
        "retrieved_segment_ids": segment_ids,
        "history_txt_path": str(hist_path),
        "response_model": response_model,
        "response_time_sec": response_time,
        "predicted_answer_raw": pred_raw,
        "predicted_choice": derive_choice(pred_raw, options),
        "judge_model": judge_model,
        "judge_mode": mode,
        "judge": judge,
    }


def main() -> None:
    args = parse_args()
    args.data_schema = DatasetSchema(load_yaml(args.config).get("dataset"))
    runtime_cfg = load_runtime_config(args.runtime_config)
    response_settings = api_settings("response", runtime_cfg, args.response_model, "gpt-5-mini")
    judge_settings = api_settings("judge", runtime_cfg, args.judge_model, "gpt-4.1-mini")
    response_client = build_client(response_settings)
    judge_client = build_client(judge_settings)
    prompt_template = load_prompt(args.response_prompt_file)
    judge_prompt = None if args.judge_mode == "none" else Path(args.judge_prompt_file).read_text(encoding="utf-8").strip()
    qpaths = discover_query_files(args)
    if not qpaths:
        raise ValueError("no query json found")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_dir = out_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers or 1))
    thread_local = threading.local()

    def clients():
        rc = getattr(thread_local, "response_client", None)
        jc = getattr(thread_local, "judge_client", None)
        if rc is None:
            rc = build_client(response_settings)
            thread_local.response_client = rc
        if jc is None:
            jc = build_client(judge_settings)
            thread_local.judge_client = jc
        return rc, jc

    ordered: Dict[int, Dict[str, Any]] = {}
    def run(i: int, p: Path) -> Dict[str, Any]:
        rc, jc = clients()
        try:
            item = process_one(p, args, rc, jc, response_settings["model"], judge_settings["model"], prompt_template, judge_prompt, history_dir)
            print(f"[response][{i}/{len(qpaths)}] sample={item['sample_id']} ok")
            return item
        except Exception as exc:
            sid = p.stem.replace("query_", "")
            print(f"[response][{i}/{len(qpaths)}] sample={sid} error={exc}")
            return {"sample_id": sid, "query_json": str(p), "error": str(exc)}

    if workers <= 1:
        for i, p in enumerate(tqdm(qpaths, total=len(qpaths), desc="Response", unit="sample"), start=1):
            ordered[i] = run(i, p)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run, i, p): i for i, p in enumerate(qpaths, start=1)}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Response", unit="sample"):
                ordered[futs[fut]] = fut.result()
    items = [ordered[i] for i in sorted(ordered)]
    judged = [x for x in items if isinstance(x.get("judge"), dict) and x["judge"].get("is_correct") is not None]
    correct = [x for x in judged if bool(x["judge"].get("is_correct"))]
    times = [float(x.get("response_time_sec", 0.0)) for x in items if isinstance(x.get("response_time_sec"), (int, float))]
    by_type: Dict[str, Dict[str, Any]] = {}
    for x in items:
        qt = str(x.get("question_type") or "unknown")
        row = by_type.setdefault(qt, {"total": 0, "judged": 0, "correct": 0, "acc": 0.0})
        row["total"] += 1
        if isinstance(x.get("judge"), dict) and x["judge"].get("is_correct") is not None:
            row["judged"] += 1
            row["correct"] += int(bool(x["judge"].get("is_correct")))
    for row in by_type.values():
        row["acc"] = row["correct"] / row["judged"] if row["judged"] else 0.0
    summary = {
        "total": len(items),
        "ok_items": len([x for x in items if "error" not in x]),
        "error_items": len([x for x in items if "error" in x]),
        "response_time_avg_sec": sum(times) / len(times) if times else 0.0,
        "judge_accuracy": len(correct) / len(judged) if judged else 0.0,
        "by_question_type": by_type,
    }
    out = {"run_started_at": datetime.now().isoformat(timespec="seconds"), "response_model": response_settings["model"], "judge_model": judge_settings["model"], "summary": summary, "items": items}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[response] saved: {args.output_json}")
    print(f"[response] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
