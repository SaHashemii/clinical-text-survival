#!/usr/bin/env python3
"""
Run the clinical text survival pipeline end to end.

Pipeline:
  clinical CSV + template -> clinical sentences
  clinical sentences + CONCH -> one .pt embedding file per patient
  embedding files + labels -> Cox survival training and C-index
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clinical sentence, CONCH embedding, and survival training pipeline.")
    parser.add_argument("--clinical", type=Path, required=True, help="Input clinical CSV.")
    parser.add_argument("--labels", type=Path, required=True, help="Input survival labels CSV.")
    parser.add_argument("--templates", type=Path, required=True, help="Clinical sentence template YAML.")
    parser.add_argument("--encoder", type=Path, required=True, help="CONCH encoder config YAML.")
    parser.add_argument("--experiment", type=Path, required=True, help="Clinical embedding experiment config YAML.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Pipeline output directory.")
    parser.add_argument("--fold", type=int, default=None, help="Optional single fold to train.")
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse existing patient .pt files instead of regenerating CONCH embeddings.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a CLI path relative to the repository root unless absolute."""
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install this repository with `pip install -e .`.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of YAML file: {path}")
    return data


def write_yaml(payload: dict[str, Any], path: str | Path) -> None:
    """Write a YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install this repository with `pip install -e .`.") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def run_step(name: str, command: list[str]) -> None:
    """Run one subprocess step and keep its terminal progress visible."""
    print()
    print(f"=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_generated_data_config(labels: Path, clinical: Path, embeddings_dir: Path, output_path: Path) -> None:
    """Write a data config compatible with existing experiment configs."""
    labels_abs = str(labels.resolve())
    clinical_abs = str(clinical.resolve())
    embeddings_abs = str(embeddings_dir.resolve())
    payload = {
        "data": {
            "root": str(REPO_ROOT),
            "labels": {
                "default": labels_abs,
                "with_urolife": labels_abs,
                "without_urolife": labels_abs,
            },
            "label_name": "default",
            "clinical_tabular": clinical_abs,
            "clinical_embeddings": {
                "default": embeddings_abs,
                "conch5": embeddings_abs,
            },
            "clinical_embedding_name": "default",
        }
    }
    write_yaml(payload, output_path)


def print_final_summary(output_dir: Path) -> None:
    """Print final C-index summary when training produced summary.json."""
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        print()
        print(f"Training finished, but no summary file was found at {summary_path}")
        return

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    mean_ci = summary.get("mean_c_index")
    std_ci = summary.get("std_c_index")
    folds = summary.get("folds")
    print()
    print("=== Final result ===")
    if mean_ci is not None:
        print(f"Mean C-index: {mean_ci:.4f}")
    if std_ci is not None:
        print(f"Std C-index: {std_ci:.4f}")
    if folds is not None:
        print(f"Folds: {folds}")
    print(f"Results written to: {output_dir}")


def main() -> None:
    args = parse_args()
    clinical = resolve_path(args.clinical)
    labels = resolve_path(args.labels)
    templates = resolve_path(args.templates)
    encoder = resolve_path(args.encoder)
    experiment = resolve_path(args.experiment)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences_jsonl = output_dir / "clinical_sentences.jsonl"
    sentences_csv = output_dir / "clinical_sentences.csv"
    embeddings_dir = output_dir / "clinical_embeddings"
    generated_data_config = output_dir / "generated_data_config.yaml"

    run_step(
        "Step 1/4: render clinical sentences",
        [
            sys.executable,
            str(SCRIPTS_DIR / "make_clinical_sentences.py"),
            "--clinical",
            str(clinical),
            "--templates",
            str(templates),
            "--output-jsonl",
            str(sentences_jsonl),
            "--output-csv",
            str(sentences_csv),
        ],
    )

    conch_command = [
        sys.executable,
        str(SCRIPTS_DIR / "make_conch_clinical_embeddings.py"),
        "--sentences",
        str(sentences_jsonl),
        "--encoder",
        str(encoder),
        "--output-dir",
        str(embeddings_dir),
    ]
    if not args.reuse_embeddings:
        conch_command.append("--overwrite")
    run_step("Step 2/4: encode sentences with CONCH", conch_command)

    print()
    print("=== Step 3/4: write generated training data config ===", flush=True)
    write_generated_data_config(labels, clinical, embeddings_dir, generated_data_config)
    print(f"Wrote {generated_data_config}", flush=True)

    train_command = [
        sys.executable,
        str(SCRIPTS_DIR / "train_clinical.py"),
        "--experiment",
        str(experiment),
        "--data",
        str(generated_data_config),
        "--output-dir",
        str(output_dir),
    ]
    if args.fold is not None:
        train_command.extend(["--fold", str(args.fold)])
    run_step("Step 4/4: train survival model", train_command)
    print_final_summary(output_dir)


if __name__ == "__main__":
    main()
