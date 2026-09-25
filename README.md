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

## Model status

| Component | Status |
|---|---|
| Swin-Base encoder (frozen) | implemented |
| Linear projection 1024 → 768 | implemented |
| **BioBART-v2-base decoder** | **implemented — this is the current baseline** |
| ClinicalT5 decoder | planned for Stage 2; ships Flax weights only, needs `from_flax=True` |
| Concept (CUI) token branch | not implemented; `models.concept_encoder` is reserved and read by no code |
| CUI classifier (diagnostic) | implemented as a standalone experiment |

The decoder is set by `models.decoder` in `config.yaml` and nothing in the code
is BioBART-specific — any seq2seq model with hidden size 768 works unchanged.
A guard raises a readable error if the decoder's hidden size disagrees with
`train.decoder_dim`.

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
├── fetch_subset.py         # optional: tiny real-data slice for a smoke test
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

Train the BioBART decoder using the extracted visual features. Only a linear
projection (1024 → 768) and the decoder are trainable; the Swin encoder stays
frozen and is never loaded at this stage.

For a short run on a seeded random subset — useful for smoke-testing a new
machine or comparing configurations without a full pass:

```bash
python decoder_training.py --limit 1000
```

Subset checkpoints are written to `checkpoints/limit1000/`, so they can never
overwrite a full run's `best.pt`. Training is resume-capable: `--resume auto`
continues from `last.pt`, which matters on Kaggle's 9-hour session limit.

**Checkpoint size.** Each checkpoint stores optimizer and scheduler state as well
as weights -- ~1.66 GB for BioBART-base -- and that size is set by the model, not
the dataset. `save_every_epoch` is therefore `false`: only `best.pt` and `last.pt`
are kept, ~3.1 GB total instead of ~19 GB for ten epochs. `last.pt` is still
written every epoch, so `--resume auto` is unaffected. Setting it to `true` on
Kaggle will overrun the 20 GB working directory mid-run.

---

### Step 4 — Evaluate

```bash
python evaluate.py --ckpt best.pt --split test
```

Generates captions with beam search and scores them with BLEU-1..4, ROUGE-1/2/L
and BERTScore. `--ckpt` is required; a bare name is resolved against
`checkpoints_dir`, so `best.pt`, `limit1000/best.pt` and an absolute path all
work. Add `--no-bertscore` to skip the ~1.4 GB `roberta-large` download.

Outputs land in `<checkpoints_dir>/eval/`. Read `samples_<split>.txt` before the
metrics -- a low BLEU with degenerate text is a different problem from a low
BLEU with fluent but wrong text, and only the samples distinguish them.

---

## Pipeline validation

The full path -- raw images to scored captions -- has been executed end to end on
real ROCOv2 data using a deliberately tiny slice (10 train / 8 validation / 8
test). `fetch_subset.py` builds that slice by reading only row group 0 of one
shard per split over ranged HTTP, ~82 MB instead of the mirror's ~17 GB:

```bash
python fetch_subset.py --root D:/rrg_smoke
RRG_ROOT=D:/rrg_smoke python load_Dataset.py
RRG_ROOT=D:/rrg_smoke python extract_features.py
RRG_ROOT=D:/rrg_smoke python decoder_training.py
RRG_ROOT=D:/rrg_smoke python evaluate.py --ckpt best.pt --split test --no-bertscore
```

The shards keep the mirror's exact schema, so every script runs unmodified. Use a
separate root: ten-row shards sitting where a real download belongs is how toy
data ends up silently training a real run.

What this established: images decode, Swin emits `[N, 49, 1024]`, the 1024 -> 768
projection feeds the decoder without shape errors, validation loss fell
monotonically 12.14 -> 9.48 across ten epochs, checkpoints round-trip, beam-search
generation runs, and BLEU/ROUGE compute.

What it does **not** establish is model quality. Ten examples cannot teach a
~140M-parameter decoder radiology grammar, so BLEU-4 is ~0.0007 and the captions
are word salad. That is the expected result of a working pipeline at this data
scale, not evidence about the architecture. The degeneracy check reported
`unique_ratio: 1.0` -- undertrained, but not mode-collapsed.

---

## Tests

```bash
pytest
```

109 tests, CPU-only, no dataset or GPU required. They cover the config loader and
its environment overrides, CUI parsing and the multi-label metric math, caption
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

| Stage | Status |
|---|---|
| Dataset pre-flight (`load_Dataset.py`) | complete; column names validated against `config.yaml` |
| Feature extraction (`extract_features.py`) | complete; overwrite guard and isolated dry runs |
| Decoder training (`decoder_training.py`) | complete; resume-capable, early stopping, warmup clamp for short runs |
| Evaluation (`evaluate.py`) | complete; BLEU-1..4, ROUGE-1/2/L, BERTScore, degeneracy report |
| End-to-end validation | done on a 26-row real-data slice — see **Pipeline validation** |
| Test suite and CI | 109 tests, CPU-only, no dataset required |
| **Full-scale GPU training** | **pending — no trained checkpoint or reportable metrics yet** |

The architecture is implemented and the pipeline is proven to run. The remaining
work is compute, not code: feature extraction over all 79,793 images, then the
decoder training pass.

---

## Future Improvements

* Concept-guided report generation -- `models.concept_encoder` is reserved in
  `config.yaml` but read by no code; the CUI branch is not implemented
* ClinicalT5 decoder (Stage 2; ships Flax weights, needs `from_flax=True`)
* CIDEr evaluation
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




