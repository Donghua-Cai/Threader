from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .store import SQLiteIndex


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    an = float(np.linalg.norm(av))
    bn = float(np.linalg.norm(bv))
    if an <= 0 or bn <= 0:
        return 0.0
    return float(np.dot(av, bv) / (an * bn))


def topk_mean(values: Iterable[float], k: int) -> Optional[float]:
    xs = sorted([float(v) for v in values], reverse=True)
    if not xs:
        return None
    kk = min(max(1, int(k)), len(xs))
    return float(sum(xs[:kk]) / kk)


def rank_map(ranked_ids: Sequence[str]) -> Dict[str, int]:
    return {str(sid): i for i, sid in enumerate(ranked_ids, start=1)}


def minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (float(v) - lo) / (hi - lo) for k, v in scores.items()}


def ordered_union(left: Sequence[str], right: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for seq in (left, right):
        for sid in seq:
            s = str(sid).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def segment_embedding_rank(
    store: SQLiteIndex,
    q_vec: np.ndarray,
    weights: Dict[str, float],
) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    wu = float(weights.get("user", 0.0))
    wa = float(weights.get("assistant", 0.0))
    wf = float(weights.get("full", 0.0))
    for sid, vu, va, vf, _ in store.iter_segments_with_embeddings():
        score = wu * cosine(q_vec, vu) + wa * cosine(q_vec, va) + wf * cosine(q_vec, vf)
        scored.append((sid, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def message_role_split_segment_scores(
    store: SQLiteIndex,
    candidate_ids: Sequence[str],
    q_vec: np.ndarray,
    *,
    topk: int = 2,
    weight_user: float = 0.7,
    weight_assistant: float = 0.3,
) -> Dict[str, float]:
    turns_by_segment = store.get_pair_turns_by_segment_ids(candidate_ids)
    out: Dict[str, float] = {}
    for sid, turns in turns_by_segment.items():
        msg_scores: List[float] = []
        for turn in turns:
            if str(turn.get("text_user") or "").strip():
                msg_scores.append(float(weight_user) * cosine(q_vec, turn["emb_user"]))
            if str(turn.get("text_assistant") or "").strip():
                msg_scores.append(float(weight_assistant) * cosine(q_vec, turn["emb_assistant"]))
        agg = topk_mean(msg_scores, topk)
        if agg is not None:
            out[str(sid)] = float(agg)
    return out


def prefusion_rank(
    candidate_ids: Sequence[str],
    embedding_scores: Dict[str, float],
    bm25_scores: Dict[str, float],
    *,
    embedding_weight: float = 1.0,
    bm25_weight: float = 0.65,
) -> List[Tuple[str, float]]:
    # Rank-based fusion: w_emb / rank_emb + w_bm25 / rank_bm25 inside the same candidate pool.
    ids = [str(x) for x in candidate_ids]
    emb_ranked = sorted(
        [sid for sid in ids if sid in embedding_scores],
        key=lambda sid: float(embedding_scores.get(sid, 0.0)),
        reverse=True,
    )
    bm25_ranked = sorted(ids, key=lambda sid: float(bm25_scores.get(sid, 0.0)), reverse=True)
    fused: Dict[str, float] = {sid: 0.0 for sid in ids}
    for rank, sid in enumerate(emb_ranked, start=1):
        fused[sid] += float(embedding_weight) / float(rank)
    for rank, sid in enumerate(bm25_ranked, start=1):
        fused[sid] += float(bm25_weight) / float(rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def rerank_segments(store: SQLiteIndex, reranker: Any, query: str, candidate_ids: Sequence[str]) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    if reranker is None or not getattr(reranker, "available", False):
        return [], {}
    rows = store.get_segments_by_ids(candidate_ids)
    by_id = {str(r["segment_id"]): r for r in rows}
    ordered_ids: List[str] = []
    passages: List[str] = []
    for sid in candidate_ids:
        row = by_id.get(str(sid))
        if not row:
            continue
        text = str(row.get("text_full") or row.get("text_user") or row.get("text_assistant") or "").strip()
        if text:
            ordered_ids.append(str(sid))
            passages.append(text)
    scores = reranker.score(query, passages) if passages else []
    score_map = {sid: float(score) for sid, score in zip(ordered_ids, scores)}
    return sorted(score_map.items(), key=lambda x: x[1], reverse=True), score_map


def hybrid_final_rank(base_scores: Dict[str, float], rerank_scores: Dict[str, float], candidate_ids: Sequence[str], alpha: float) -> List[Tuple[str, float]]:
    if not rerank_scores:
        return sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
    n_base = minmax_norm({sid: float(base_scores.get(sid, 0.0)) for sid in candidate_ids})
    n_rerank = minmax_norm(rerank_scores)
    final = {
        str(sid): float(alpha) * float(n_base.get(str(sid), 0.0)) + (1.0 - float(alpha)) * float(n_rerank.get(str(sid), 0.0))
        for sid in candidate_ids
    }
    return sorted(final.items(), key=lambda x: x[1], reverse=True)

