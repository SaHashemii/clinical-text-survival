# Clinical Text Survival Pipeline

This repository trains a clinical-only survival model from structured clinical
variables.

The intended workflow is:

```text
clinical CSV + labels CSV + sentence template
  -> clinical sentences
  -> CONCH text embeddings
  -> Cox survival model
  -> C-index
```

## Repository Setup

```bash
git clone <this-repo-url>
cd clinical_unimodal_survival
conda create -n clinical-text-survival python=3.10
conda activate clinical-text-survival
pip install -e .
```

## CONCH Setup

```bash
git clone https://github.com/mahmoodlab/CONCH.git ../CONCH
pip install -e ../CONCH
mkdir -p ../CONCH/checkpoints/conch
```

Download the CONCH weights and place them at:

```text
../CONCH/checkpoints/conch/pytorch_model.bin
```

Check the setup:

```bash
python -c "import clinical_survival; print('repo ok')"
python -c "from conch.open_clip_custom import create_model_from_pretrained; print('conch ok')"
```

If you already have a Conda environment for CONCH, activate it and run
`pip install -e .` from this repository.

## Reproducible Environment Files

The repository includes `requirements.txt` for the core package dependencies.

If you want to record the exact Conda environment you used, export it from the
active environment:

```bash
conda env export --from-history > environment.yml
```

Before committing `environment.yml`, remove any machine-specific line like:

```yaml
prefix: /Users/your-name/miniconda3/envs/env-name
```

You can also save exact pip package versions:

```bash
pip freeze > requirements.lock.txt
```

Before committing `requirements.lock.txt`, remove local editable paths such as:

```text
-e /path/to/CONCH
clinical-unimodal-survival @ file:///...
```

CONCH itself should stay as a separate setup step because it is a separate
repository and its weights require separate access.

## Input Files

You need:

```text
clinical CSV
labels CSV
sentence template YAML
CONCH encoder config YAML
experiment config YAML
```

The clinical CSV must contain one row per patient and an ID column:

```text
sample_id
```

The default sentence template expects these clinical columns:

```text
age
sex
smoking
tumor
stage
substage
grade
reTUR
LVI
variant
EORTC
no_instillations
BRS
```

Missing clinical values can be encoded as:

```text
-1
-1.0
```

The labels CSV must contain or be convertible to:

```text
sample_id
Time
Event
```

Common label names such as `case_id`, `time`, `event`, and progression columns
are normalized automatically.

## Sentence Template

Clinical variables are converted into natural-language sentences using:

```text
configs/text/clinical_sentence_templates.yaml
```

Example numeric variable:

```yaml
age:
  column: age
  type: numeric
  template: "The patient is {value} years old."
  missing: "The patient's age is unknown."
  decimals: 0
```

Example binary variable:

```yaml
smoking:
  column: smoking
  type: binary
  template: "The patient has a history of smoking."
  negative_template: "The patient has no history of smoking."
  missing: "The patient's smoking history is unknown."
  positive_values: ["yes", "Yes", "YES"]
  negative_values: ["no", "No", "NO"]
```

You can edit the template wording or add new variables, but the clinical CSV
columns and template rules must match.

## End-To-End Pipeline

Run the full clinical text survival pipeline with one command:

```bash
python3 scripts/run_clinical_text_survival.py \
  --clinical data/clinical.csv \
  --labels data/labels.csv \
  --templates configs/text/clinical_sentence_templates.yaml \
  --encoder configs/encoders/conch.example.yaml \
  --experiment configs/experiments/clinical_embedding.yaml \
  --output-dir outputs/clinical_text_conch
```

This command:

```text
1. Render clinical variables into patient-level sentences.
2. Encode those sentences with CONCH.
3. Save one clinical embedding .pt file per patient.
4. Train the Cox survival model with cross-validation.
5. Write the final C-index.
```

The terminal shows progress messages and progress bars for the major steps.

Expected output:

```text
outputs/clinical_text_conch/
  clinical_sentences.jsonl
  clinical_sentences.csv
  clinical_embeddings/
  generated_data_config.yaml
  fold_assignments.csv
  fold_*/
  results_per_fold.csv
  summary.json
```

The final mean C-index is saved in:

```text
outputs/clinical_text_conch/summary.json
```

The per-fold C-index values are saved in:

```text
outputs/clinical_text_conch/results_per_fold.csv
```

By default, CONCH embeddings are regenerated. To reuse existing `.pt` files in
the output directory, add:

```bash
--reuse-embeddings
```

## Current Step-By-Step Commands

The end-to-end runner calls these individual commands internally. You can also
run them manually.

### 1. Make Clinical Sentences

```bash
python3 scripts/make_clinical_sentences.py \
  --clinical data/clinical.csv \
  --templates configs/text/clinical_sentence_templates.yaml \
  --output-jsonl outputs/clinical_sentences.jsonl \
  --output-csv outputs/clinical_sentences.csv
```

This writes:

```text
outputs/clinical_sentences.jsonl
outputs/clinical_sentences.csv
```

JSONL example:

```json
{"sample_id": "PATIENT_001", "sentences": ["The patient is 72 years old.", "The patient is male."], "features": ["age", "sex"]}
```

### 2. Make CONCH Clinical Embeddings

```bash
python3 scripts/make_conch_clinical_embeddings.py \
  --sentences outputs/clinical_sentences.jsonl \
  --encoder configs/encoders/conch.example.yaml \
  --output-dir outputs/clinical_embeddings_conch
```

This writes one `.pt` file per patient:

```text
outputs/clinical_embeddings_conch/
  PATIENT_001.pt
  PATIENT_002.pt
  ...
```

Each file contains:

```python
{
    "sample_id": "...",
    "sentences": [...],
    "features": [...],
    "embeddings": tensor
}
```

### 3. Train Survival Model

Create a local data config:

```bash
cp configs/data/example.yaml configs/data/local.yaml
```

Edit `configs/data/local.yaml` so it points to your labels and CONCH embedding folder.

Example:

```yaml
data:
  root: .
  labels:
    default: data/labels.csv
  label_name: default
  clinical_embeddings:
    conch_sentences: outputs/clinical_embeddings_conch
  clinical_embedding_name: conch_sentences
```

Train:

```bash
python3 scripts/train_clinical.py \
  --experiment configs/experiments/clinical_embedding.yaml \
  --data configs/data/local.yaml \
  --output-dir outputs/clinical_text_conch
```

## Optional: Tabular Clinical Model

You can also train directly on tabular clinical variables without CONCH:

```bash
python3 scripts/train_clinical.py \
  --experiment configs/experiments/clinical_tabular.yaml \
  --data configs/data/local.yaml
```

## Training Outputs

A training run writes:

```text
fold_assignments.csv
fold_*/history.csv
fold_*/train_risk_scores.csv
fold_*/test_risk_scores.csv
fold_*/summary.json
fold_*/metadata.json
fold_*/model.pt
results_per_fold.csv
test_risk_scores_all_folds.csv
kaplan_meier_by_risk.png
summary.json
```

## Notes

- CONCH is not included in this repository.
- CONCH weights are not included in this repository.
- Keep local paths in local config files.
- Generated outputs are written under `outputs/`.
