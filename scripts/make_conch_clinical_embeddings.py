#!/usr/bin/env python3
"""
Encode patient-level clinical sentences with the CONCH text encoder.

Input JSONL records are produced by scripts/make_clinical_sentences.py:
  {"sample_id": "PATIENT_001", "sentences": [...], "features": [...]}

Each output .pt file contains:
  {
    "sample_id": str,
    "sentences": list[str],
    "features": list[str],
    "embeddings": torch.Tensor,  # [num_sentences, conch_text_dim]
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CONCH text embeddings from clinical sentence JSONL.")
    parser.add_argument("--sentences", type=Path, required=True, help="Input patient-level sentence JSONL.")
    parser.add_argument("--encoder", type=Path, required=True, help="CONCH encoder config YAML.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for one .pt file per patient.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing patient .pt files.")
    return parser.parse_args()


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


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load encoder config. Install with `pip install pyyaml`.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level of YAML config: {path}")
    return data


def import_torch():
    """Import PyTorch only when embedding generation is actually run."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to create CONCH embeddings. Install this repo's dependencies first.") from exc
    return torch


def resolve_repo_path(value: str | Path | None) -> Path | None:
    """Resolve config paths relative to the repository root unless absolute."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_sentence_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate patient sentence records from JSONL."""
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", "")).strip()
            sentences = record.get("sentences")
            if not sample_id:
                raise ValueError(f"Line {line_no} is missing a non-empty sample_id.")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id in sentence JSONL: {sample_id}")
            if not isinstance(sentences, list) or not sentences:
                raise ValueError(f"Line {line_no} for {sample_id} must contain a non-empty sentences list.")
            if not all(isinstance(sentence, str) and sentence.strip() for sentence in sentences):
                raise ValueError(f"Line {line_no} for {sample_id} contains an empty or non-string sentence.")
            features = record.get("features", [])
            if features and len(features) != len(sentences):
                raise ValueError(
                    f"Line {line_no} for {sample_id} has {len(features)} features "
                    f"but {len(sentences)} sentences."
                )
            seen_ids.add(sample_id)
            records.append(
                {
                    "sample_id": sample_id,
                    "sentences": sentences,
                    "features": features,
                }
            )
    if not records:
        raise ValueError(f"No sentence records found in {path}")
    return records


def import_conch(repo_path: Path | None):
    """Import CONCH, optionally from a local clone path."""
    if repo_path is not None:
        if not repo_path.is_dir():
            raise FileNotFoundError(f"CONCH repo_path does not exist: {repo_path}")
        sys.path.insert(0, str(repo_path))

    try:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    except ImportError as exc:
        raise ImportError(
            "Could not import CONCH. Clone CONCH next to this repo and install it with "
            "`pip install -e ../CONCH`, or set conch.repo_path in the encoder config."
        ) from exc
    return create_model_from_pretrained, get_tokenizer, tokenize


def encode_text_batches(
    model,
    tokenize_fn,
    tokenizer,
    sentences: list[str],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode a patient's sentences in batches and return a CPU tensor."""
    torch = import_torch()
    chunks = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        tokens = tokenize_fn(texts=batch, tokenizer=tokenizer).to(device)
        with torch.inference_mode():
            embeddings = model.encode_text(tokens)
        chunks.append(embeddings.detach().cpu().float())
    return torch.cat(chunks, dim=0)


def main() -> None:
    args = parse_args()
    print(f"[1/5] Loading encoder config: {args.encoder}", flush=True)
    torch = import_torch()
    encoder_cfg_full = load_yaml(args.encoder)
    encoder_cfg = encoder_cfg_full.get("conch")
    if not isinstance(encoder_cfg, dict):
        raise ValueError("Encoder config must contain a top-level `conch` mapping.")

    repo_path = resolve_repo_path(encoder_cfg.get("repo_path"))
    checkpoint_path = resolve_repo_path(encoder_cfg.get("checkpoint_path"))
    if checkpoint_path is None:
        raise ValueError("Encoder config must define conch.checkpoint_path.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CONCH checkpoint does not exist: {checkpoint_path}")

    model_name = str(encoder_cfg.get("model_name", "conch_ViT-B-16"))
    device_name = str(encoder_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.", file=sys.stderr)
        device_name = "cpu"
    device = torch.device(device_name)
    batch_size = int(encoder_cfg.get("batch_size", 64))
    if batch_size <= 0:
        raise ValueError("conch.batch_size must be a positive integer.")

    print("[2/5] Importing CONCH", flush=True)
    create_model_from_pretrained, get_tokenizer, tokenize_fn = import_conch(repo_path)
    print(f"[3/5] Loading CONCH model: {model_name}", flush=True)
    model, _ = create_model_from_pretrained(model_name, checkpoint_path=str(checkpoint_path))
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer()

    print(f"[4/5] Loading sentence JSONL: {args.sentences}", flush=True)
    records = load_sentence_records(args.sentences)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[5/5] Encoding {len(records)} patients on {device}", flush=True)
    written = 0
    skipped = 0
    for record in progress(records, desc="Encoding patients", unit="patient"):
        sample_id = record["sample_id"]
        output_path = args.output_dir / f"{sample_id}.pt"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        embeddings = encode_text_batches(
            model,
            tokenize_fn,
            tokenizer,
            record["sentences"],
            batch_size=batch_size,
            device=device,
        )
        torch.save(
            {
                "sample_id": sample_id,
                "sentences": record["sentences"],
                "features": record["features"],
                "embeddings": embeddings,
            },
            output_path,
        )
        written += 1

    print(f"Wrote {written} patient embedding files to {args.output_dir}")
    if skipped:
        print(f"Skipped {skipped} existing files; pass --overwrite to regenerate them.")


if __name__ == "__main__":
    main()
