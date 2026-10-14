from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .config import load_yaml


def load_runtime_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    return load_yaml(p) if p.exists() else {}


def api_settings(kind: str, runtime_cfg: Dict[str, Any], model_override: Optional[str], fallback_model: str) -> Dict[str, str]:
    block = runtime_cfg.get("openai") if isinstance(runtime_cfg.get("openai"), dict) else {}
    default = block.get("default") if isinstance(block.get("default"), dict) else {}
    scoped = block.get(kind) if isinstance(block.get(kind), dict) else {}
    kind_upper = kind.upper()
    api_key = str(
        os.getenv(f"{kind_upper}_API_KEY")
        or scoped.get("api_key")
        or os.getenv("OPENAI_API_KEY")
        or default.get("api_key")
        or ""
    ).strip()
    base_url = str(
        os.getenv(f"{kind_upper}_BASE_URL")
        or scoped.get("base_url")
        or os.getenv("OPENAI_BASE_URL")
        or default.get("base_url")
        or ""
    ).strip()
    model = str(model_override or scoped.get("model") or default.get("model") or fallback_model).strip()
    return {"api_key": api_key, "base_url": base_url, "model": model}


def build_client(settings: Dict[str, str]):
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai is required for response generation") from exc
    kwargs: Dict[str, Any] = {}
    if settings.get("api_key"):
        kwargs["api_key"] = settings["api_key"]
    if settings.get("base_url"):
        kwargs["base_url"] = settings["base_url"]
    return OpenAI(**kwargs)
