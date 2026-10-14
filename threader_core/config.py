from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyyaml is required to load YAML config files") from exc
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def deep_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    cur: Dict[str, Any] = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    for key, value in overrides.items():
        if value is not None:
            deep_set(out, key, value)
    return out
