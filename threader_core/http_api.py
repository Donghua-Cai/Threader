from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib import error, request


def resolve_api_config(
    kind: str,
    api_url: Optional[str] = None,
    api_token: Optional[str] = None,
    api_token_env: Optional[str] = None,
) -> Dict[str, str]:
    prefix = (kind or "").strip().upper()
    token = str(api_token or "").strip()
    if not token and api_token_env:
        token = str(os.getenv(str(api_token_env), "") or "").strip()
    if not token:
        for key in [
            f"{prefix}_API_KEY",
            f"{prefix}_API_TOKEN",
            f"HF_{prefix}_API_TOKEN",
            f"HUGGINGFACE_{prefix}_API_TOKEN",
            "HF_API_TOKEN",
            "HUGGINGFACE_API_TOKEN",
            "HF_TOKEN",
        ]:
            token = str(os.getenv(key, "") or "").strip()
            if token:
                break

    url = str(api_url or "").strip()
    if not url:
        for key in [
            f"{prefix}_API_URL",
            f"HF_{prefix}_API_URL",
            f"HUGGINGFACE_{prefix}_API_URL",
        ]:
            url = str(os.getenv(key, "") or "").strip()
            if url:
                break
    return {"api_url": url, "api_token": token}


def post_json(url: str, payload: Any, api_token: str = "", timeout_sec: float = 60.0) -> Any:
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=float(timeout_sec)) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            body = resp.read().decode(charset)
    except error.HTTPError as exc:
        charset = exc.headers.get_content_charset() if exc.headers else "utf-8"
        raw = exc.read().decode(charset or "utf-8", errors="replace")
        raise RuntimeError(f"API request failed ({exc.code}) for {url}: {raw}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"API request failed for {url}: {exc}") from exc
    try:
        return json.loads(body)
    except Exception as exc:
        raise RuntimeError(f"API response is not valid JSON for {url}: {body[:300]}") from exc


def parse_embedding_response(obj: Any) -> List[List[float]]:
    if isinstance(obj, dict):
        data = obj.get("data")
        if isinstance(data, list):
            out: List[List[float]] = []
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("embedding"), list):
                    out.append([float(x) for x in item["embedding"]])
            if out:
                return out
        if isinstance(obj.get("embeddings"), list):
            rows = obj["embeddings"]
            if rows and isinstance(rows[0], list):
                return [[float(x) for x in row] for row in rows]
        if isinstance(obj.get("embedding"), list):
            return [[float(x) for x in obj["embedding"]]]
    if isinstance(obj, list):
        if not obj:
            return []
        if _is_number(obj[0]):
            return [[float(x) for x in obj]]
        if isinstance(obj[0], list):
            return [[float(x) for x in row] for row in obj]
    raise RuntimeError(f"unsupported embedding response format: {type(obj).__name__}")


def parse_rerank_response(obj: Any, expected_len: int) -> List[float]:
    if isinstance(obj, dict):
        for key in ("results", "data", "rankings"):
            if isinstance(obj.get(key), list):
                return _parse_rank_items(obj[key], expected_len)
        if isinstance(obj.get("scores"), list):
            return [float(x) for x in obj["scores"]][:expected_len]
    if isinstance(obj, list):
        if not obj:
            return []
        if _is_number(obj[0]):
            return [float(x) for x in obj][:expected_len]
        return _parse_rank_items(obj, expected_len)
    raise RuntimeError(f"unsupported rerank response format: {type(obj).__name__}")


def _parse_rank_items(items: List[Any], expected_len: int) -> List[float]:
    scores = [0.0 for _ in range(expected_len)]
    filled = 0
    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if idx is None:
            idx = item.get("document_index")
        if idx is None:
            idx = pos
        try:
            idx_int = int(idx)
        except Exception:
            continue
        if idx_int < 0 or idx_int >= expected_len:
            continue
        score = item.get("score")
        if score is None:
            score = item.get("relevance_score")
        if score is None:
            continue
        scores[idx_int] = float(score)
        filled += 1
    if filled == 0:
        raise RuntimeError("rerank response did not contain usable scores")
    return scores


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
