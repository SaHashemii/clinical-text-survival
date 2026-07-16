#!/usr/bin/env python3
"""
Render structured clinical covariates into deterministic patient sentences.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clinical_survival.text.templates import (
    load_template_config,
    read_clinical_csv,
    render_patient_sentences,
    validate_template_config,
    write_sentences_csv,
    write_sentences_jsonl,
)


def progress(iterable, **kwargs):
    """Return a tqdm progress iterator when tqdm is installed."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        total = kwargs.get("total")
        desc = kwargs.get("desc", "Progress")
        unit = kwargs.get("unit", "item")
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        if not total:
            return iterable

        def simple_progress():
            width = 30
            update_every = max(1, total // 20)
            for idx, item in enumerate(iterable, start=1):
                if idx == 1 or idx == total or idx % update_every == 0:
                    filled = int(width * idx / total)
                    bar = "#" * filled + "-" * (width - filled)
                    print(f"\r{desc}: [{bar}] {idx}/{total} {unit}", end="", file=sys.stderr, flush=True)
                yield item
            print(file=sys.stderr)

        return simple_progress()
    return tqdm(iterable, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create natural-language clinical sentences from a CSV.")
    parser.add_argument("--clinical", type=Path, required=True, help="Input clinical CSV.")
    parser.add_argument("--templates", type=Path, required=True, help="Clinical sentence template YAML.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Patient-level JSONL output for text encoding.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional long CSV output for inspection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[1/4] Loading template config: {args.templates}", flush=True)
    config = load_template_config(args.templates)
    print(f"[2/4] Loading clinical CSV: {args.clinical}", flush=True)
    rows = read_clinical_csv(args.clinical)
    print(f"[3/4] Rendering sentences for {len(rows)} patients", flush=True)
    if rows:
        validate_template_config(config, list(rows[0].keys()))
    id_column = config.get("settings", {}).get("id_column", "sample_id")
    payloads = []
    seen_ids = set()
    for row_idx, row in enumerate(progress(rows, desc="Rendering patients", unit="patient"), start=1):
        sample_id = str(row.get(id_column, "")).strip()
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

    print("[4/4] Writing output files", flush=True)
    write_sentences_jsonl(payloads, args.output_jsonl)
    if args.output_csv is not None:
        write_sentences_csv(payloads, args.output_csv)

    n_sentences = sum(len(payload["sentences"]) for payload in payloads)
    print(f"Wrote {len(payloads)} patients and {n_sentences} sentences to {args.output_jsonl}")
    if args.output_csv is not None:
        print(f"Wrote inspection CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
