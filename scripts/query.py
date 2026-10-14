#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threader_core.config import load_yaml
from threader_core.data import discover_sample_files, extract_question_meta, find_segment_sample_path, index_question_files, load_question_meta, load_sample_id_file, load_segment_sample, sample_id_from_path
from threader_core.embeddings import Embedder
from threader_core.reranker import CrossEncoderReranker
from threader_core.schema import DatasetSchema
from threader_core.scoring import hybrid_final_rank, message_role_split_segment_scores, ordered_union, prefusion_rank, rerank_segments, segment_embedding_rank
from threader_core.store import SQLiteIndex

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None, unit=None):
        return iterable if iterable is not None else range(total or 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VSM baseline retrieval and export query_*.json files.")
    p.add_argument("--config", default=str(ROOT / "config" / "baseline.yaml"))
    p.add_argument("--index-dir", required=True)
    p.add_argument("--question-dir", default=None, help="Optional QA directory. If omitted, QA fields are read from segment samples.")
    p.add_argument("--segment-dir", required=True)
    p.add_argument("--question-type", default="ALL")
    p.add_argument("--sample-id", default=None)
    p.add_argument("--sample-ids-file", default=None)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--sample-offset", type=int, default=0)
    p.add_argument("--exclude-abstention", type=int, choices=[0, 1], default=None)
    p.add_argument("--final-topk", type=int, default=None)
    p.add_argument("--ranked-source", choices=["hybrid", "base", "rerank"], default="hybrid")
    p.add_argument("--output-json", required=True)
    p.add_argument("--query-output-dir", required=True)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def discover_samples(args: argparse.Namespace, schema: Optional[DatasetSchema] = None) -> List[str]:
    if args.sample_id:
        return [str(args.sample_id)]
    if args.sample_ids_file:
        sample_ids = load_sample_id_file(args.sample_ids_file)
    else:
        sample_root = args.question_dir or args.segment_dir
        sample_ids = sorted({
            sample_id_from_path(path)
            for path in discover_sample_files(sample_root, question_type=args.question_type, schema=schema)
        })
    if args.exclude_abstention is not None and int(args.exclude_abstention) == 1:
        filtered: List[str] = []
        for sid in sample_ids:
            seg_path = find_segment_sample_path(args.segment_dir, sid, None)
            if seg_path is not None and "abstention" in seg_path.parts:
                continue
            qpath = index_question_files(args.question_dir).get(sid) if args.question_dir else None
            if qpath and ("abstention" in qpath.parts or "_abs" in qpath.stem):
                continue
            filtered.append(sid)
        sample_ids = filtered
    sample_ids = sample_ids[int(args.sample_offset or 0):]
    if args.sample_limit is not None:
        sample_ids = sample_ids[: int(args.sample_limit)]
    return sample_ids


def resolve_db(index_dir: Path, sample_id: str, question_type: Optional[str]) -> Path:
    candidates = []
    if question_type and question_type.upper() != "ALL":
        candidates.append(index_dir / question_type / f"{sample_id}.db")
    candidates.append(index_dir / f"{sample_id}.db")
    candidates.extend(sorted(index_dir.rglob(f"{sample_id}.db")))
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"missing db for sample {sample_id} under {index_dir}")


def run_one(sample_id: str, args: argparse.Namespace, cfg: Dict[str, Any], embedder: Embedder, reranker: CrossEncoderReranker) -> Dict[str, Any]:
    t0 = time.perf_counter()
    question_dir = Path(args.question_dir) if args.question_dir else None
    segment_dir = Path(args.segment_dir)
    index_dir = Path(args.index_dir)
    schema = DatasetSchema(cfg.get("dataset"))
    qmeta = load_question_meta(question_dir, sample_id, schema=schema) if question_dir else {}
    if not qmeta.get("question"):
        seg_for_meta = load_segment_sample(segment_dir, sample_id, args.question_type, schema=schema)
        qmeta = extract_question_meta(seg_for_meta or {}, sample_id=sample_id, schema=schema)
    question = str(qmeta.get("question") or "").strip()
    question_type = str(qmeta.get("question_type") or args.question_type or "").strip()
    question_date = qmeta.get("question_date")
    # Retrieval uses the raw question, matching the no-rewrite baseline.
    # The response stage still prepends question_date before calling the answer model.
    q_text = question
    if not q_text:
        raise ValueError(f"question text not found for sample {sample_id}")

    db_path = resolve_db(index_dir, sample_id, question_type)
    q_vec = embedder.embed_text(q_text)
    router = cfg.get("router", {})
    rerank_cfg = cfg.get("reranker", {})
    weights = router.get("query_channel_weights", router.get("query_channel_weights_default", {}))
    final_topk = int(args.final_topk or router.get("final_topk", 12))
    segment_sim_topk = int(router.get("segment_sim_topk", 40))
    bm25_topk = int(router.get("bm25_topk", 40))
    turn_score_topk = int(router.get("turn_score_topk", 2))
    bm25_weight = float(router.get("pre_fusion_weight_bm25", 0.65))
    embedding_weight = float(router.get("pre_fusion_weight_embedding", 1.0))
    alpha = float(rerank_cfg.get("hybrid_alpha", router.get("final_hybrid_alpha", 0.3)))
    rerank_pre_topn = int(rerank_cfg.get("pre_topn", 20))
    message_weight_user = float(router.get("message_weight_user", 0.7))
    message_weight_assistant = float(router.get("message_weight_assistant", 0.3))
    use_turn_embedding = bool(router.get("use_turn_embedding", True))

    store = SQLiteIndex(db_path, emb_dim=embedder.dim)
    try:
        segment_ranked = segment_embedding_rank(store, q_vec, weights)
        bm25_ranked = store.bm25_rank(q_text)
        embedding_candidate_ids = [sid for sid, _ in segment_ranked[:segment_sim_topk]]
        bm25_candidate_ids = [sid for sid, _ in bm25_ranked[:bm25_topk]]
        pre_candidates = ordered_union(embedding_candidate_ids, bm25_candidate_ids)
        segment_score_map = {sid: float(score) for sid, score in segment_ranked}
        bm25_score_map = {sid: float(score) for sid, score in bm25_ranked}

        if use_turn_embedding:
            turn_score_map = message_role_split_segment_scores(
                store,
                pre_candidates,
                q_vec,
                topk=turn_score_topk,
                weight_user=message_weight_user,
                weight_assistant=message_weight_assistant,
            )
            embedding_scores_for_fusion = {str(sid): float(score) for sid, score in turn_score_map.items()}
        else:
            embedding_scores_for_fusion = {sid: float(segment_score_map.get(sid, 0.0)) for sid in pre_candidates}

        base_ranked = prefusion_rank(pre_candidates, embedding_scores_for_fusion, bm25_score_map, embedding_weight=embedding_weight, bm25_weight=bm25_weight)
        rerank_pool_ids = [sid for sid, _ in base_ranked[:rerank_pre_topn]]
        rerank_ranked, rerank_score_map = rerank_segments(store, reranker, q_text, rerank_pool_ids)
        base_score_for_final = {sid: float(score) for sid, score in base_ranked if sid in set(rerank_pool_ids)}
        final_ranked = hybrid_final_rank(base_score_for_final, rerank_score_map, rerank_pool_ids, alpha=alpha)
    finally:
        store.close()

    ranked_lookup = {
        "base": [sid for sid, _ in base_ranked],
        "rerank": [sid for sid, _ in rerank_ranked],
        "hybrid": [sid for sid, _ in final_ranked],
    }
    selected = ranked_lookup[str(args.ranked_source)][:final_topk]
    return {
        "sample_id": sample_id,
        "question_id": qmeta.get("question_id"),
        "dataset": qmeta.get("dataset"),
        "db_path": str(db_path),
        "question": question,
        "question_type": question_type,
        "question_date": question_date,
        "num_segments": len(selected),
        "segment_ids": selected,
        "retrieved_segment_ids": selected,
        "base_all_segment_ids": ranked_lookup["base"],
        "base_segment_ids": ranked_lookup["base"][:final_topk],
        "rerank_all_segment_ids": ranked_lookup["rerank"],
        "rerank_segment_ids": ranked_lookup["rerank"][:final_topk],
        "hybrid_all_segment_ids": ranked_lookup["hybrid"],
        "hybrid_segment_ids": ranked_lookup["hybrid"][:final_topk],
        "embedding_top20_segment_ids": [sid for sid, _ in segment_ranked[:20]],
        "bm25_top20_segment_ids": [sid for sid, _ in bm25_ranked[:20]],
        "answer": qmeta.get("answer"),
        "answer_options": qmeta.get("answer_options", []),
        "correct_answer_label": qmeta.get("correct_answer_label"),
        "query_time_sec": float(time.perf_counter() - t0),
    }


def export_query_files(items: Sequence[Dict[str, Any]], output_dir: Path, report_path: Path, ranked_source: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        payload = {
            "query_id": item["sample_id"],
            "sample_id": item["sample_id"],
            "question_id": item.get("question_id"),
            "dataset": item.get("dataset"),
            "db_path": item["db_path"],
            "question": item.get("question"),
            "question_type": item.get("question_type"),
            "question_date": item.get("question_date"),
            "answer": item.get("answer"),
            "answer_options": item.get("answer_options", []),
            "correct_answer_label": item.get("correct_answer_label"),
            "num_segments": len(item.get("segment_ids") or []),
            "segment_ids": item.get("segment_ids") or [],
            "retrieved_segment_ids": item.get("segment_ids") or [],
            "source_report_json": str(report_path),
            "ranked_source": ranked_source,
        }
        (output_dir / f"query_{item['sample_id']}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"source_report_json": str(report_path), "ranked_source": ranked_source, "exported": len(items)}
    (output_dir / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    schema = DatasetSchema(cfg.get("dataset"))
    if args.exclude_abstention is not None:
        cfg.setdefault("dataset", {})["exclude_abstention"] = bool(args.exclude_abstention)
    else:
        args.exclude_abstention = int(bool(cfg.get("dataset", {}).get("exclude_abstention", False)))
    emb_cfg = cfg.get("embedding", {})
    embedder = Embedder(
        emb_cfg.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        int(emb_cfg.get("batch_size", 64)),
        int(emb_cfg.get("dim_fallback", 384)),
        backend=str(emb_cfg.get("backend", "sentence_transformers")),
        api_url=str(emb_cfg.get("api_url", "")),
        api_token=str(emb_cfg.get("api_token", "")),
        api_token_env=str(emb_cfg.get("api_token_env", "")),
        api_timeout_sec=float(emb_cfg.get("api_timeout_sec", 60.0)),
    )
    rr_cfg = cfg.get("reranker", {})
    reranker = CrossEncoderReranker(
        rr_cfg.get("model_name", "BAAI/bge-reranker-base"),
        int(rr_cfg.get("batch_size", 1)),
        bool(rr_cfg.get("enabled", True)),
        backend=str(rr_cfg.get("backend", "cross_encoder")),
        api_url=str(rr_cfg.get("api_url", "")),
        api_token=str(rr_cfg.get("api_token", "")),
        api_token_env=str(rr_cfg.get("api_token_env", "")),
        api_timeout_sec=float(rr_cfg.get("api_timeout_sec", 60.0)),
    )

    sample_ids = discover_samples(args, schema=schema)
    workers = max(1, int(args.workers or 1))
    print(f"[query] samples={len(sample_ids)} ranked_source={args.ranked_source} workers={workers}")
    ordered: Dict[str, Dict[str, Any]] = {}
    if workers <= 1:
        for sid in tqdm(sample_ids, total=len(sample_ids), desc="Query", unit="sample"):
            ordered[sid] = run_one(sid, args, cfg, embedder, reranker)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one, sid, args, cfg, embedder, reranker): sid for sid in sample_ids}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Query", unit="sample"):
                ordered[futs[fut]] = fut.result()
    items = [ordered[sid] for sid in sample_ids if sid in ordered]
    report_path = Path(args.output_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    router = cfg.get("router", {})
    rerank_cfg = cfg.get("reranker", {})
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_questions": len(items),
        "runtime": {"workers": workers, "embedding_model": emb_cfg.get("model_name"), "reranker_model": rr_cfg.get("model_name")},
        "hparams": {
            "ranked_source": args.ranked_source,
            "final_topk": int(args.final_topk or router.get("final_topk", 12)),
            "segment_sim_topk": int(router.get("segment_sim_topk", 40)),
            "bm25_topk": int(router.get("bm25_topk", 40)),
            "rerank_pre_topn": int(rerank_cfg.get("pre_topn", 20)),
            "turn_score_topk": int(router.get("turn_score_topk", 2)),
            "use_turn_embedding": int(bool(router.get("use_turn_embedding", True))),
            "message_role_split_score": 1,
            "message_role_score_weighted": 1,
            "message_weight_user": float(router.get("message_weight_user", 0.7)),
            "message_weight_assistant": float(router.get("message_weight_assistant", 0.3)),
            "pre_fusion_weight_embedding": float(router.get("pre_fusion_weight_embedding", 1.0)),
            "pre_fusion_weight_bm25": float(router.get("pre_fusion_weight_bm25", 0.65)),
            "prefusion_mode": "rank",
            "final_hybrid_alpha": float(rerank_cfg.get("hybrid_alpha", 0.3)),
            "query_channel_weights": dict(router.get("query_channel_weights", {})),
            "exclude_abstention": bool(args.exclude_abstention),
        },
    }
    out = {"summary": summary, "questions": items}
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    export_query_files(items, Path(args.query_output_dir), report_path, args.ranked_source)
    print(f"[query] report={report_path} exported={len(items)} query_dir={args.query_output_dir}")


if __name__ == "__main__":
    main()
