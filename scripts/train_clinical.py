#!/usr/bin/env python3
"""
Train clinical unimodal Cox survival models with cross-validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clinical_survival.data.clinical import (
    fit_clinical_preprocessor,
    load_clinical_embeddings,
    load_clinical_table,
    stack_clinical_embeddings,
    transform_clinical_table,
)
from clinical_survival.data.labels import load_labels
from clinical_survival.data.splits import prepare_fold_split, summarize_fold_split
from clinical_survival.models.clinical_cox import ClinicalCoxModel
from clinical_survival.training.artifacts import ensure_dir, save_checkpoint, write_history, write_json
from clinical_survival.training.clinical_trainer import evaluate_clinical_cox, train_clinical_cox
from clinical_survival.training.cross_validation import make_fold_assignments
from clinical_survival.training.plots import write_kaplan_meier_plot
from clinical_survival.utils.config import load_yaml, materialize_data_config, resolve_path
from clinical_survival.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a clinical-only Cox survival model.")
    parser.add_argument("--experiment", type=Path, required=True, help="Clinical experiment config YAML.")
    parser.add_argument("--data", type=Path, required=True, help="Data config YAML.")
    parser.add_argument("--fold-assignments", type=Path, default=None, help="Optional existing fold_assignments.csv.")
    parser.add_argument("--fold", type=int, default=None, help="Train only one fold.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    return parser.parse_args()


def _to_tensor(values, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def _label_tensors(labels: pd.DataFrame, sample_ids: list[str], device: torch.device):
    time = _to_tensor(labels.loc[sample_ids, "Time"].values.astype(np.float32), device)
    event = _to_tensor(labels.loc[sample_ids, "Event"].values.astype(np.float32), device)
    return time, event


def _load_or_make_folds(sample_ids, labels, cv_cfg, output_dir, fold_assignments_path):
    if fold_assignments_path is not None:
        return pd.read_csv(fold_assignments_path)
    output_path = output_dir / "fold_assignments.csv"
    if output_path.is_file():
        return pd.read_csv(output_path)
    folds = make_fold_assignments(sample_ids, labels, int(cv_cfg.get("n_splits", 5)), int(cv_cfg.get("seed", 42)))
    folds.to_csv(output_path, index=False)
    return folds


def _train_kwargs(train_cfg: dict, device: torch.device) -> dict:
    return {
        "device": device,
        "epochs": int(train_cfg.get("epochs", 300)),
        "patience": int(train_cfg.get("patience", 40)),
        "batch_size": int(train_cfg.get("batch_size", 64)),
        "min_events_per_batch": int(train_cfg.get("min_events_per_batch", 3)),
        "training_style": train_cfg.get("training_style", "full_batch"),
        "lr": float(train_cfg.get("lr", 2e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 1e-5)),
        "grad_clip": float(train_cfg.get("grad_clip", 5.0)),
    }


def train_clinical_fold(exp_cfg, labels, clinical_data, split, clinical_source, device, train_cfg):
    model_cfg = exp_cfg.get("model", {})
    if clinical_source == "embedding":
        token_count = max(int(emb.shape[0]) for emb in clinical_data.values())
        clinical_dim = int(next(iter(clinical_data.values())).shape[1])
        x_fit = stack_clinical_embeddings(clinical_data, split.fit_ids, token_count, clinical_dim)
        x_train = stack_clinical_embeddings(clinical_data, split.train_ids, token_count, clinical_dim)
        x_val = stack_clinical_embeddings(clinical_data, split.val_ids, token_count, clinical_dim)
        x_test = stack_clinical_embeddings(clinical_data, split.test_ids, token_count, clinical_dim)
        model = ClinicalCoxModel(
            clinical_source="embedding",
            clinical_dim=clinical_dim,
            clinical_token_count=token_count,
            clinical_emb_dim=int(model_cfg.get("projection_dim", 512)),
            clinical_token_hidden_dim=int(model_cfg.get("attention_hidden_dim", 128)),
            clinical_token_out_dim=int(model_cfg.get("attention_hidden_dim", 128)),
            clinical_dropout=float(model_cfg.get("dropout", 0.30)),
            clinical_activation=model_cfg.get("activation", "selu"),
            clinical_pooling=model_cfg.get("pooling"),
            clinical_projection_dim=int(model_cfg.get("projection_dim", 512)),
            clinical_attention_hidden_dim=int(model_cfg.get("attention_hidden_dim", 128)),
            head_hidden_dims=model_cfg.get("hidden_dims", [128]),
            head_dropout=float(model_cfg.get("dropout", 0.30)),
            head_activation=model_cfg.get("activation", "selu"),
        ).to(device)
        extra = {
            "clinical_source": clinical_source,
            "clinical_token_count": token_count,
            "clinical_dim": clinical_dim,
            "pooling": model_cfg.get("pooling"),
            "projection_dim": int(model_cfg.get("projection_dim", 512)),
        }
    else:
        clinical_index = clinical_data.set_index("sample_id")
        clin_fit = clinical_index.loc[split.fit_ids].reset_index()
        clin_train = clinical_index.loc[split.train_ids].reset_index()
        clin_val = clinical_index.loc[split.val_ids].reset_index()
        clin_test = clinical_index.loc[split.test_ids].reset_index()
        meta = fit_clinical_preprocessor(clin_fit)
        x_fit = transform_clinical_table(clin_fit, meta)
        x_train = transform_clinical_table(clin_train, meta)
        x_val = transform_clinical_table(clin_val, meta)
        x_test = transform_clinical_table(clin_test, meta)
        model = ClinicalCoxModel(
            clinical_source="tabular",
            clinical_dim=x_train.shape[1],
            clinical_hidden_dims=model_cfg.get("hidden_dims", [128]),
            clinical_emb_dim=int(model_cfg.get("projection_dim", 128)),
            clinical_dropout=float(model_cfg.get("dropout", 0.30)),
            clinical_activation=model_cfg.get("activation", "selu"),
            head_hidden_dims=[],
        ).to(device)
        extra = {"clinical_source": clinical_source, "clinical_preprocessing": meta, "clinical_dim": x_train.shape[1]}

    fit_tensors = (_to_tensor(x_fit, device), *_label_tensors(labels, split.fit_ids, device))
    val_tensors = (_to_tensor(x_val, device), *_label_tensors(labels, split.val_ids, device))
    train_tensors = (_to_tensor(x_train, device), *_label_tensors(labels, split.train_ids, device))
    test_tensors = (_to_tensor(x_test, device), *_label_tensors(labels, split.test_ids, device))
    model, history = train_clinical_cox(model, fit_tensors, val_tensors=val_tensors, **_train_kwargs(train_cfg, device))
    return model, history, train_tensors, test_tensors, extra


def main() -> None:
    args = parse_args()
    exp_cfg = load_yaml(args.experiment)
    data_cfg_full = load_yaml(args.data)
    data_cfg_raw = data_cfg_full.get("data", {})
    exp_info = exp_cfg.get("experiment", {})
    exp_data_cfg = exp_cfg.get("data", {})
    data_cfg = materialize_data_config(data_cfg_raw, exp_data_cfg)
    cv_cfg = exp_cfg.get("cv", {})
    train_cfg = exp_cfg.get("training", {})
    clinical_source = exp_data_cfg.get("clinical_source", "embedding")

    output_dir = ensure_dir(args.output_dir or resolve_path(REPO_ROOT, exp_info.get("output_dir", "outputs/clinical_unimodal")))
    device_name = str(train_cfg.get("device", "cpu"))
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.", file=sys.stderr)
        device_name = "cpu"
    device = torch.device(device_name)
    seed = int(cv_cfg.get("seed", 42))
    set_seed(seed)
    data_root = Path(data_cfg["root"]).expanduser()
    labels = load_labels(resolve_path(data_root, data_cfg["labels"]))

    if clinical_source == "embedding":
        source = load_clinical_embeddings(resolve_path(data_root, data_cfg["clinical_embeddings"]))
        sample_ids = sorted(set(labels.index.astype(str)) & set(source))
    elif clinical_source == "tabular":
        source = load_clinical_table(resolve_path(data_root, data_cfg["clinical_tabular"]))
        sample_ids = sorted(set(labels.index.astype(str)) & set(source["sample_id"].astype(str)))
    else:
        raise ValueError(f"Unsupported clinical_source: {clinical_source}")
    labels = labels.loc[sample_ids]

    fold_assignments = _load_or_make_folds(sample_ids, labels, cv_cfg, output_dir, args.fold_assignments)
    folds_to_run = [args.fold] if args.fold is not None else list(range(int(cv_cfg.get("n_splits", 5))))
    results = []
    test_risk_tables = []
    for fold in folds_to_run:
        split = prepare_fold_split(sample_ids, labels, fold_assignments, fold, seed=seed + fold, val_size=float(cv_cfg.get("val_size", 0.20)))
        set_seed(seed + fold)
        model, history, train_data, test_data, extra = train_clinical_fold(exp_cfg, labels, source, split, clinical_source, device, train_cfg)
        train_ci, train_risk = evaluate_clinical_cox(model, train_data, split.train_ids)
        test_ci, test_risk = evaluate_clinical_cox(model, test_data, split.test_ids)

        fold_dir = ensure_dir(output_dir / f"fold_{fold}")
        write_history(history, fold_dir / "history.csv")
        train_risk.to_csv(fold_dir / "train_risk_scores.csv", index=False)
        test_risk.to_csv(fold_dir / "test_risk_scores.csv", index=False)
        test_risk_tables.append(test_risk.assign(fold=fold))
        summary = {**summarize_fold_split(labels, split), "c_index": test_ci, "train_c_index": train_ci, "model_type": "clinical_unimodal"}
        write_json(summary, fold_dir / "summary.json")
        write_json(extra, fold_dir / "metadata.json")
        save_checkpoint(fold_dir / "model.pt", model, summary, extra={"metadata": extra, "experiment_config": exp_cfg})
        print(f"[fold {fold}] test_c_index={test_ci:.4f} train_c_index={train_ci:.4f}")
        results.append(summary)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results_per_fold.csv", index=False)
    aggregate = {
        "experiment": exp_info.get("name", args.experiment.stem),
        "folds": len(results),
        "mean_c_index": float(results_df["c_index"].mean()) if not results_df.empty else None,
        "std_c_index": float(results_df["c_index"].std(ddof=0)) if not results_df.empty else None,
    }
    if test_risk_tables:
        all_test_risks = pd.concat(test_risk_tables, ignore_index=True)
        all_test_risks.to_csv(output_dir / "test_risk_scores_all_folds.csv", index=False)
        aggregate["kaplan_meier"] = write_kaplan_meier_plot(
            all_test_risks,
            output_dir / "kaplan_meier_by_risk.png",
            title=str(aggregate["experiment"]),
        )
    write_json(aggregate, output_dir / "summary.json")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
