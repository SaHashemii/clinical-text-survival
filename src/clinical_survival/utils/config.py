"""
Configuration helpers for clinical survival experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of YAML config: {path}")
    return data


def resolve_path(root: str | Path, value: str | Path | None) -> Path | None:
    """Resolve a path relative to an explicit root."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(root).expanduser() / path


def _select_config_value(data_cfg: dict[str, Any], exp_data_cfg: dict[str, Any], key: str, selector_key: str) -> Any:
    value = data_cfg.get(key)
    if not isinstance(value, dict):
        return value
    selected = exp_data_cfg.get(selector_key, data_cfg.get(selector_key))
    if selected is None:
        raise ValueError(f"Data config {key} is a mapping; define data.{selector_key}.")
    if selected not in value:
        options = ", ".join(sorted(str(option) for option in value))
        raise ValueError(f"Unknown {selector_key}={selected!r}. Available values: {options}")
    return value[selected]


def materialize_data_config(data_cfg: dict[str, Any], exp_data_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve named labels and clinical embedding choices into concrete paths."""
    exp_data_cfg = exp_data_cfg or {}
    resolved = dict(data_cfg)
    resolved["labels"] = _select_config_value(data_cfg, exp_data_cfg, "labels", "label_name")
    resolved["clinical_embeddings"] = _select_config_value(
        data_cfg,
        exp_data_cfg,
        "clinical_embeddings",
        "clinical_embedding_name",
    )
    return resolved

