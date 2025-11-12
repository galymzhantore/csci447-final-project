from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def merge_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if value is None:
            continue
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = merge_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_label_schema(schema: str | Mapping[str, Any]) -> Dict[str, int]:
    if isinstance(schema, Mapping):
        return {str(k): int(v) for k, v in schema.items()}
    if schema == "3class":
        return {"poor": 0, "moderate": 1, "good": 2}
    try:
        parsed = json.loads(schema)
    except json.JSONDecodeError as err:  # pragma: no cover - sanity guard
        raise ValueError(
            "LABEL_SCHEMA must be '3class' or a JSON object string"
        ) from err
    if not isinstance(parsed, dict):
        raise ValueError("Parsed label schema is not a dict")
    return {str(k): int(v) for k, v in parsed.items()}


__all__ = ["load_config", "merge_overrides", "parse_label_schema"]
