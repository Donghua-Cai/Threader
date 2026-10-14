from __future__ import annotations

from typing import List, Sequence

from .http_api import parse_rerank_response, post_json, resolve_api_config


class CrossEncoderReranker:
    """Reranker wrapper: local CrossEncoder, HTTP API, or disabled."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 1,
        enabled: bool = True,
        backend: str = "cross_encoder",
        api_url: str = "",
        api_token: str = "",
        api_token_env: str = "",
        api_timeout_sec: float = 60.0,
    ):
        self.model_name = str(model_name)
        self.batch_size = int(batch_size)
        self.enabled = bool(enabled)
        self.backend = str(backend or "cross_encoder").strip().lower()
        self.available = False
        self.model = None
        self.api_url = ""
        self.api_token = ""
        self.api_timeout_sec = float(api_timeout_sec)
        if not self.enabled or self.backend == "none":
            self.backend = "none"
            return
        if self.backend in {"cross_encoder", "local"}:
            self._init_cross_encoder()
        elif self.backend in {"api", "huggingface_api"}:
            self._init_api(api_url=api_url, api_token=api_token, api_token_env=api_token_env)
        else:
            raise ValueError(f"unsupported reranker backend: {backend}")

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        if not self.enabled or not self.available:
            return [0.0 for _ in passages]
        if self.backend in {"api", "huggingface_api"}:
            return self._score_api(str(query or ""), passages)
        pairs = [[str(query or ""), str(p or "")] for p in passages]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(x) for x in scores]

    def _init_cross_encoder(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("sentence-transformers is required for local reranker backend") from exc
        self.model = CrossEncoder(self.model_name)
        self.available = True
        self.backend = "cross_encoder"

    def _init_api(self, api_url: str, api_token: str, api_token_env: str) -> None:
        resolved = resolve_api_config(
            kind="RERANKER",
            api_url=api_url,
            api_token=api_token,
            api_token_env=api_token_env,
        )
        self.api_url = resolved["api_url"]
        self.api_token = resolved["api_token"]
        if not self.api_url:
            raise ValueError("reranker.api_url is required when reranker.backend is an API backend")
        self.available = True

    def _score_api(self, query: str, passages: Sequence[str]) -> List[float]:
        docs = [str(p or "") for p in passages]
        last_error: Exception | None = None
        for payload in self._rerank_payload_candidates(query, docs):
            try:
                obj = post_json(self.api_url, payload, api_token=self.api_token, timeout_sec=self.api_timeout_sec)
                return parse_rerank_response(obj, expected_len=len(docs))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"reranker API request failed for {self.api_url}: {last_error}") from last_error

    def _rerank_payload_candidates(self, query: str, passages: Sequence[str]) -> List[object]:
        docs = list(passages)
        return [
            {"query": query, "texts": docs},
            {"query": query, "documents": docs},
            {"inputs": {"query": query, "texts": docs}},
            {"inputs": {"query": query, "documents": docs}},
            {"model": self.model_name, "query": query, "documents": docs},
        ]
