"""
Template-based clinical sentence rendering.

This module turns one row of structured clinical covariates into deterministic
natural-language sentences. The renderer is intentionally strict by default so
template/data mismatches fail before CONCH embeddings are created.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


VALID_TYPES = {"numeric", "categorical", "binary"}


def load_template_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML template config."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load sentence templates. Install with `pip install pyyaml`.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping at top level of template config: {path}")
    return config


def read_clinical_csv(path: str | Path) -> list[dict[str, str]]:
    """Read clinical rows from CSV as dictionaries."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Clinical CSV has no header: {path}")
        return list(reader)


def _as_key(value: Any) -> str:
    return str(value).strip()


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _numeric_equal(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def is_missing_value(value: Any, missing_values: list[Any]) -> bool:
    """Return True when a value matches configured missing sentinels."""
    if value is None:
        return True
    if _is_nan(value):
        return True
    value_key = _as_key(value)
    for missing in missing_values:
        if value == missing:
            return True
        if value_key == _as_key(missing):
            return True
        if _numeric_equal(value, missing):
            return True
    return False


def validate_template_config(config: dict[str, Any], columns: list[str]) -> None:
    """Validate template structure against available clinical CSV columns."""
    settings = config.get("settings", {})
    templates = config.get("templates")
    if not isinstance(settings, dict):
        raise ValueError("Template config `settings` must be a mapping.")
    if not isinstance(templates, dict) or not templates:
        raise ValueError("Template config must define a non-empty `templates` mapping.")

    id_column = settings.get("id_column", "sample_id")
    if id_column not in columns:
        raise ValueError(f"ID column {id_column!r} is missing from clinical CSV.")

    column_set = set(columns)
    for name, spec in templates.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Template {name!r} must be a mapping.")
        for key in ("column", "type", "missing"):
            if key not in spec:
                raise ValueError(f"Template {name!r} is missing required key: {key}")
        if spec["column"] not in column_set:
            raise ValueError(f"Template {name!r} references missing clinical column: {spec['column']!r}")
        if spec["type"] not in VALID_TYPES:
            raise ValueError(f"Template {name!r} has unsupported type {spec['type']!r}; expected {sorted(VALID_TYPES)}")

        if spec["type"] == "binary":
            for key in ("template", "negative_template", "positive_values", "negative_values"):
                if key not in spec:
                    raise ValueError(f"Binary template {name!r} is missing required key: {key}")
            if not isinstance(spec["positive_values"], list) or not isinstance(spec["negative_values"], list):
                raise ValueError(f"Binary template {name!r} positive_values and negative_values must be lists.")
        else:
            if "template" not in spec:
                raise ValueError(f"Template {name!r} is missing required key: template")
            if "{value}" not in spec["template"]:
                raise ValueError(f"Template {name!r} template must contain {{value}}.")

        if "value_map" in spec and not isinstance(spec["value_map"], dict):
            raise ValueError(f"Template {name!r} value_map must be a mapping.")


def _format_numeric(value: Any, spec: dict[str, Any]) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {spec['column']!r} expected numeric value, got {value!r}.") from exc
    decimals = int(spec.get("decimals", 2))
    return f"{number:.{decimals}f}"


def _render_nonmissing(value: Any, spec: dict[str, Any]) -> str:
    value_key = _as_key(value)
    if spec["type"] == "binary":
        positive = {_as_key(item) for item in spec["positive_values"]}
        negative = {_as_key(item) for item in spec["negative_values"]}
        if value_key in positive:
            return str(spec["template"])
        if value_key in negative:
            return str(spec["negative_template"])
        raise ValueError(
            f"Unexpected binary value {value!r} for column {spec['column']!r}. "
            f"Allowed positive={sorted(positive)}, negative={sorted(negative)}."
        )

    if "value_map" in spec:
        value_map = {_as_key(key): mapped for key, mapped in spec["value_map"].items()}
        if value_key not in value_map:
            raise ValueError(
                f"Unexpected categorical value {value!r} for column {spec['column']!r}. "
                f"Allowed values={sorted(value_map)}."
            )
        rendered_value = str(value_map[value_key])
    elif spec["type"] == "numeric":
        rendered_value = _format_numeric(value, spec)
    else:
        rendered_value = value_key

    return str(spec["template"]).format(value=rendered_value)


def render_patient_sentences(row: dict[str, Any], config: dict[str, Any]) -> list[dict[str, str]]:
    """Render sentence records for one patient row."""
    settings = config.get("settings", {})
    missing_values = settings.get("missing_values", ["", -1, "-1"])
    include_missing = bool(settings.get("include_missing", True))

    rendered: list[dict[str, str]] = []
    for feature, spec in config["templates"].items():
        value = row.get(spec["column"])
        if is_missing_value(value, missing_values):
            if include_missing:
                rendered.append({"feature": feature, "sentence": str(spec["missing"])})
            continue
        rendered.append({"feature": feature, "sentence": _render_nonmissing(value, spec)})
    return rendered


def render_clinical_sentences(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Render grouped patient-level sentence payloads."""
    if not rows:
        return []
    validate_template_config(config, list(rows[0].keys()))
    id_column = config.get("settings", {}).get("id_column", "sample_id")

    payloads = []
    seen_ids: set[str] = set()
    for row_idx, row in enumerate(rows, start=1):
        sample_id = _as_key(row.get(id_column, ""))
        if not sample_id:
            raise ValueError(f"Row {row_idx} has an empty sample ID in column {id_column!r}.")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id found: {sample_id}")
        seen_ids.add(sample_id)
        sentence_records = render_patient_sentences(row, config)
        payloads.append(
            {
                "sample_id": sample_id,
                "sentences": [record["sentence"] for record in sentence_records],
                "features": [record["feature"] for record in sentence_records],
            }
        )
    return payloads


def write_sentences_jsonl(payloads: list[dict[str, Any]], path: str | Path) -> None:
    """Write patient-level sentence payloads as JSONL for text encoders."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_sentences_csv(payloads: list[dict[str, Any]], path: str | Path) -> None:
    """Write one sentence per row for human inspection."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sentence_index", "feature", "sentence"])
        writer.writeheader()
        for payload in payloads:
            for idx, (feature, sentence) in enumerate(zip(payload["features"], payload["sentences"])):
                writer.writerow(
                    {
                        "sample_id": payload["sample_id"],
                        "sentence_index": idx,
                        "feature": feature,
                        "sentence": sentence,
                    }
                )
