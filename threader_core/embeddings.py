from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .http_api import parse_embedding_response, post_json, resolve_api_config


class Embedder:
    """Embedding backend wrapper: local SentenceTransformer or HTTP API."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        dim_fallback: int = 384,
        backend: str = "sentence_transformers",
        api_url: str = "",
        api_token: str = "",
        api_token_env: str = "",
        api_timeout_sec: float = 60.0,
    ):
        self.model_name = str(model_name)
        self.batch_size = int(batch_size)
        self.dim = int(dim_fallback)
        self.backend = str(backend or "sentence_transformers").strip().lower()
        self.model = None
        self.api_url = ""
        self.api_token = ""
        self.api_timeout_sec = float(api_timeout_sec)
        if self.backend in {"sentence_transformers", "local"}:
            self._init_sentence_transformers()
        elif self.backend in {"api", "huggingface_api", "openai_compatible"}:
            self._init_api(api_url=api_url, api_token=api_token, api_token_env=api_token_env)
        else:
            raise ValueError(f"unsupported embedding backend: {backend}")

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        batch: List[str] = [str(x or "") for x in texts]
        if not batch:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.backend in {"api", "huggingface_api", "openai_compatible"}:
            return self._embed_api_batch(batch)
        vecs = self.model.encode(
            batch,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def _init_sentence_transformers(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("sentence-transformers is required for local embedding backend") from exc
        self.model = SentenceTransformer(self.model_name)
        probe = self.model.encode(["test"], normalize_embeddings=True, show_progress_bar=False)
        self.dim = int(np.asarray(probe).shape[1])
        self.backend = "sentence_transformers"

    def _init_api(self, api_url: str, api_token: str, api_token_env: str) -> None:
        resolved = resolve_api_config(
            kind="EMBEDDING",
            api_url=api_url,
            api_token=api_token,
            api_token_env=api_token_env,
        )
        self.api_url = resolved["api_url"]
        self.api_token = resolved["api_token"]
        if not self.api_url:
            raise ValueError("embedding.api_url is required when embedding.backend is an API backend")
        probe = self._embed_api_once(["test"])
        if probe.size == 0 or probe.shape[1] <= 0:
            raise RuntimeError(f"embedding API returned empty embeddings: {self.api_url}")
        self.dim = int(probe.shape[1])

    def _embed_api_batch(self, texts: List[str]) -> np.ndarray:
        rows: List[np.ndarray] = []
        step = max(1, int(self.batch_size))
        for start in range(0, len(texts), step):
            rows.append(self._embed_api_once(texts[start : start + step]))
        return np.vstack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)

    def _embed_api_once(self, texts: List[str]) -> np.ndarray:
        last_error: Exception | None = None
        for payload in self._embedding_payload_candidates(texts):
            try:
                obj = post_json(self.api_url, payload, api_token=self.api_token, timeout_sec=self.api_timeout_sec)
                embeddings = parse_embedding_response(obj)
                if len(embeddings) != len(texts):
                    raise RuntimeError(f"embedding response count mismatch: expected {len(texts)} got {len(embeddings)}")
                return _l2_normalize(np.asarray(embeddings, dtype=np.float32))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"embedding API request failed for {self.api_url}: {last_error}") from last_error

    def _embedding_payload_candidates(self, texts: List[str]) -> List[object]:
        url = (self.api_url or "").lower()
        batch = list(texts)
        payloads: List[object] = []
        if "/v1/embeddings" in url:
            payloads.append({"model": self.model_name, "input": batch})
            payloads.append({"input": batch})
        payloads.append({"inputs": batch})
        if len(batch) == 1:
            payloads.append({"inputs": batch[0]})
        if "/v1/embeddings" not in url:
            payloads.append({"model": self.model_name, "input": batch})
        payloads.append(batch)
        return payloads


def zero_vec(dim: int) -> np.ndarray:
    return np.zeros((int(dim),), dtype=np.float32)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n <= 0:
        return arr
    return arr / n


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.ndim == 1:
        return l2_normalize(mat)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms
