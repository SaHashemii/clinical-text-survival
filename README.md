# Clinical Unimodal Survival

Clinical-only Cox survival modeling extracted from the multimodal survival fusion repository.

This repository supports two clinical inputs:

- `embedding`: one `.pt` tensor per sample, shaped `[tokens, dim]` or `[dim]`.
- `tabular`: one CSV row per sample with `sample_id` or `case_id` plus clinical covariates.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configure Data

Copy the example data config and edit the paths:

```bash
cp configs/data/example.yaml configs/data/local.yaml
```

Labels must contain or be convertible to:

- `sample_id`
- `Time`
- `Event`

Common legacy names such as `case_id`, `time`, `event`, and progression columns are normalized automatically.

## Train

Embedding clinical model:

```bash
python scripts/train_clinical.py \
  --experiment configs/experiments/clinical_embedding.yaml \
  --data configs/data/local.yaml
```

Tabular clinical model:

```bash
python scripts/train_clinical.py \
  --experiment configs/experiments/clinical_tabular.yaml \
  --data configs/data/local.yaml
```

Train only one fold:

```bash
python scripts/train_clinical.py \
  --experiment configs/experiments/clinical_tabular.yaml \
  --data configs/data/local.yaml \
  --fold 0
```

## Outputs

Each run writes:

- `fold_assignments.csv`
- `fold_*/history.csv`
- `fold_*/train_risk_scores.csv`
- `fold_*/test_risk_scores.csv`
- `fold_*/summary.json`
- `fold_*/model.pt`
- `results_per_fold.csv`
- `test_risk_scores_all_folds.csv`
- `kaplan_meier_by_risk.png`
- `summary.json`

## Fold Assignments

To create folds separately:

```bash
python scripts/make_folds.py --data configs/data/local.yaml --output outputs/fold_assignments.csv
```
