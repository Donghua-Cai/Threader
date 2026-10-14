#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threader_core.config import load_yaml
from threader_core.data import build_segment_compact, discover_sample_files, expand_session_map, extract_pair_turns
from threader_core.embeddings import Embedder, zero_vec
from threader_core.schema import DatasetSchema
from threader_core.store import SQLiteIndex

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None, unit=None):
        return iterable if iterable is not None else range(total or 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build VSM baseline indexes: segment embeddings, BM25, and pair-turn embeddings.")
    p.add_argument("--config", default=str(ROOT / "config" / "baseline.yaml"))
    p.add_argument("--segment-dir", required=True)
    p.add_argument("--question-dir", default=None, help="Optional QA directory for question-type filtering when flat segment files have no question_type field.")
    p.add_argument("--index-dir", required=True)
    p.add_argument("--question-type", default="ALL")
    p.add_argument("--sample-id", default=None)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def build_one(sample_file: Path, segment_root: Path, index_root: Path, cfg: Dict[str, Any], overwrite: bool, embedder: Embedder) -> Dict[str, Any]:
    t0 = time.perf_counter()
    schema = DatasetSchema(cfg.get("dataset"))
    try:
        obj = schema.segment_sample(json.loads(sample_file.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise ValueError(f"{sample_file}: {exc}") from exc
    sample_id = sample_file.stem.split("_", 1)[0]
    conv_id = str(obj.get("conversation_id") or sample_id)
    rel_parent = sample_file.parent.relative_to(segment_root)
    db_dir = index_root / rel_parent
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{sample_id}.db"
    if db_path.exists() and overwrite:
        os.remove(db_path)

    store = SQLiteIndex(db_path, emb_dim=embedder.dim)
    bm25_docs: List[tuple[str, str]] = []
    segment_count = 0
    pair_turn_count = 0
    orphan_assistant = 0
    dangling_user = 0
    try:
        segments = obj.get("segments_messages") or []
        sess_map = expand_session_map(obj.get("session_ids") or [], obj.get("session_num_segments") or [])
        if len(sess_map) < len(segments):
            sess_map.extend([(None, "") for _ in range(len(segments) - len(sess_map))])

        for seg_idx, seg_msgs in enumerate(segments):
            compact = build_segment_compact(seg_msgs)
            text_user = compact["text_user"]
            text_assistant = compact["text_assistant"]
            text_full = compact["text_full"]
            emb_user = embedder.embed_text(text_user)
            emb_assistant = embedder.embed_text(text_assistant) if text_assistant.strip() else zero_vec(embedder.dim)
            emb_full = embedder.embed_text(text_full)
            session_idx, session_label = sess_map[seg_idx]
            segment_id = f"{conv_id}:{seg_idx}"
            store.insert_segment(
                {
                    "segment_id": segment_id,
                    "conv_id": conv_id,
                    "seg_idx": seg_idx,
                    "session_idx": session_idx,
                    "session_label": session_label,
                    "start_order": compact["start_order"],
                    "end_order": compact["end_order"],
                    "start_time": compact["start_time"],
                    "end_time": compact["end_time"],
                    "text_user": text_user,
                    "text_assistant": text_assistant,
                    "text_full": text_full,
                    "emb_user": emb_user,
                    "emb_assistant": emb_assistant,
                    "emb_full": emb_full,
                }
            )
            bm25_docs.append((segment_id, text_full))
            pairs, dangling, orphan = extract_pair_turns(seg_msgs)
            dangling_user += dangling
            orphan_assistant += orphan
            store.delete_turns_for_segment(segment_id)
            for turn_idx, pair in enumerate(pairs):
                store.insert_pair_turn(
                    {
                        "segment_id": segment_id,
                        "turn_idx": turn_idx,
                        "turn_order": pair.get("turn_order"),
                        "turn_time": pair.get("turn_time"),
                        "text_user": pair["text_user"],
                        "text_assistant": pair["text_assistant"],
                        "text_turn": pair["text_turn"],
                        "emb_user": embedder.embed_text(pair["text_user"]),
                        "emb_assistant": embedder.embed_text(pair["text_assistant"]),
                        "emb_turn": embedder.embed_text(pair["text_turn"]),
                    }
                )
                pair_turn_count += 1
            segment_count += 1
        store.rebuild_bm25_index(bm25_docs)
        store.commit()
    finally:
        store.close()

    return {
        "sample_id": sample_id,
        "db_path": str(db_path),
        "segments": segment_count,
        "pair_turns": pair_turn_count,
        "orphan_assistant_messages": orphan_assistant,
        "dangling_user_messages": dangling_user,
        "elapsed_sec": float(time.perf_counter() - t0),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    schema = DatasetSchema(cfg.get("dataset"))
    emb_cfg = cfg.get("embedding", {})
    embedder = Embedder(
        model_name=emb_cfg.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        batch_size=int(emb_cfg.get("batch_size", 64)),
        dim_fallback=int(emb_cfg.get("dim_fallback", 384)),
        backend=str(emb_cfg.get("backend", "sentence_transformers")),
        api_url=str(emb_cfg.get("api_url", "")),
        api_token=str(emb_cfg.get("api_token", "")),
        api_token_env=str(emb_cfg.get("api_token_env", "")),
        api_timeout_sec=float(emb_cfg.get("api_timeout_sec", 60.0)),
    )

    segment_root = Path(args.segment_dir)
    index_root = Path(args.index_dir)
    files = discover_sample_files(segment_root, question_type=args.question_type, sample_id=args.sample_id, question_dir=args.question_dir, schema=schema)
    files = files[int(args.sample_offset or 0):]
    if args.sample_limit is not None:
        files = files[: int(args.sample_limit)]
    if args.question_type and args.question_type.upper() != "ALL" and (segment_root / args.question_type).exists():
        segment_root = segment_root / args.question_type
        index_root = index_root / args.question_type
    index_root.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers or 1))
    summary_path = index_root / "build_summary.jsonl"
    if args.overwrite and summary_path.exists():
        summary_path.unlink()
    print(f"[build] samples={len(files)} index_dir={index_root} workers={workers}")
    results: List[Dict[str, Any]] = []
    if workers <= 1:
        iterable = tqdm(files, total=len(files), desc="Build", unit="sample")
        for f in iterable:
            out = build_one(f, segment_root, index_root, cfg, bool(args.overwrite), embedder)
            results.append(out)
            with summary_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(build_one, f, segment_root, index_root, cfg, bool(args.overwrite), embedder) for f in files]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Build", unit="sample"):
                out = fut.result()
                results.append(out)
                with summary_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    total_elapsed = sum(float(x.get("elapsed_sec", 0.0)) for x in results)
    runtime = {"summary_type": "runtime", "samples": len(results), "workers": workers, "total_sample_elapsed_sec": total_elapsed, "avg_sample_elapsed_sec": total_elapsed / len(results) if results else 0.0}
    with summary_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(runtime, ensure_ascii=False) + "\n")
    print(f"[build] done summary={summary_path}")


if __name__ == "__main__":
    main()
