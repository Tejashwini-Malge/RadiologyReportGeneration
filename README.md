#  Radiology Report Generation using Swin Transformer and BioBART

## Overview

Radiology Report Generation (RRG) aims to automatically generate clinically meaningful reports from medical images. This project implements an encoder-decoder architecture using a Vision Transformer encoder and a biomedical language model decoder trained on the ROCOv2 dataset.

The goal is to bridge computer vision and natural language generation by producing radiology-style captions from medical images while maintaining clinically relevant information.

---

## Features

* Swin Transformer image encoder
* BioBART report decoder
* CUI (Concept Unique Identifier) prediction module
* Frozen visual feature extraction
* HDF5 feature storage for efficient training
* End-to-end training and evaluation pipeline
* Google Colab compatible workflow

---

## Dataset

This project uses the **ROCOv2 (Radiology Objects in Context Version 2)** dataset.

Dataset splits:

| Split      |    Samples |
| ---------- | ---------: |
| Train      |     59,962 |
| Validation |      9,904 |
| Test       |      9,927 |
| **Total**  | **79,793** |

The dataset consists of radiology images paired with corresponding medical captions and UMLS Concept Unique Identifiers (CUIs).

---

## Model Architecture

```text
                Medical Image
                      │
                      ▼
        Swin Transformer Encoder
                      │
      Visual Feature Extraction
                      │
            Frozen Feature Storage
                  (HDF5 Files)
                      │
                      ▼
          BioBART Decoder
                      │
              Generated Report
```

---

## Project Structure

```text
RadiologyReportGeneration/
│
├── bootstrap_env.py        # create dirs, point HF/Torch caches off C:
├── make_config.py          # generates config.yaml
├── rrg_config.py           # config loader + RRG_* env overrides
├── config.yaml
│
├── load_Dataset.py         # step 1: pre-flight check on the parquet shards
├── extract_features.py     # step 2: frozen Swin -> .h5
├── decoder_training.py     # step 3: train the BioBART decoder
├── evaluate.py             # step 4: generate + score
├── cui_classifier.py       # side experiment: do the features carry CUI signal?
│
├── tests/                  # pytest suite (no GPU, no dataset required)
├── .github/workflows/      # CI
├── pytest.ini
├── requirements.txt
└── .gitignore
```

Generated during execution:

```text
raw/
features/
checkpoints/
cache/
```

These folders are excluded from version control.

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/RadiologyReportGeneration.git
cd RadiologyReportGeneration
```

Install dependencies

```bash
pip install -r requirements.txt
```

### Where the data lives

`config.yaml` ships with Windows paths (`D:/rrg/...`). You do **not** have to edit
it to run elsewhere -- set `RRG_ROOT` and every path moves together:

```bash
export RRG_ROOT=/content/drive/MyDrive/rrg
```

Individual paths can be overridden on their own with `RRG_ROCOV2_DIR`,
`RRG_FEATURES_DIR`, `RRG_CHECKPOINTS_DIR`, and `RRG_CACHE_ROOT`; these win over
`RRG_ROOT`. Edit `make_config.py` and re-run it if you want to change the
committed defaults -- `config.yaml` is generated, and CI checks the two agree.

---

## Workflow

### Step 0 — Environment (Windows only, optional)

```bash
python bootstrap_env.py
```

Creates the `D:/rrg` tree and persists `HF_HOME` / `TORCH_HOME` / `PIP_CACHE_DIR`
so multi-GB downloads do not land on `C:`. On Colab or Linux, set `RRG_ROOT`
instead and skip this.

---

### Step 1 — Dataset Preparation

Point `RRG_ROOT` (or `config.yaml`) at the ROCOv2 download, then:

```bash
python load_Dataset.py
```

Prints row counts, shard counts and the schema, and verifies that every column
name in `config.yaml` -- including `image_id_col` -- actually exists. Do not skip
this: a wrong column name here fails on the first batch of a multi-hour
extraction run.

---

### Step 2 — Feature Extraction

```bash
python extract_features.py
```

This extracts frozen Swin Transformer features and stores them in HDF5 format.

---

### Step 3 — Train Decoder

```bash
python decoder_training.py
```

Train the BioBART decoder using the extracted visual features.

---

### Step 4 — Evaluate

```bash
python evaluate.py
```

Evaluate generated reports on the test split.

---

## Tests

```bash
pytest
```

The suite runs on CPU with no dataset present (~40 s). It covers the config
loader and its overrides, CUI parsing and the multi-label metric math, caption
collation and label masking, checkpoint save/resume round-trips, the evaluation
metrics and degeneracy check, and -- most importantly -- the overwrite guard and
dry-run isolation in `extract_features.py`, which exist because a stray re-run
once destroyed a complete 5.8 GB `train.h5`.

CI runs it on Python 3.10-3.12 and additionally checks that `config.yaml` still
matches what `make_config.py` generates.

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Swin Transformer
* BioBART
* PyArrow
* HDF5
* NumPy
* Google Colab

---

## Current Status

* Dataset pipeline completed
* Feature extraction pipeline validated
* Decoder training pipeline implemented
* Evaluation pipeline implemented
* Test suite and CI in place
* **Full-scale GPU training pending — no trained checkpoint or metrics yet**

---

## Future Improvements

* Concept-guided report generation
* CIDEr evaluation (BLEU, ROUGE and BERTScore are implemented; beam search is
  already the default at `--beams 4`)
* RadGraph and GREEN based evaluation
* Multi-GPU training
* Clinical error analysis

---

## License

This project is intended for research and educational purposes.

The ROCOv2 dataset is distributed under its respective license. Please refer to the original dataset documentation for usage terms.

---

## Author
Tejashwini Malge




