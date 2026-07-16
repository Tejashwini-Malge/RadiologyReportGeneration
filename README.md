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
├── extract_features.py
├── decoder_training.py
├── evaluate.py
├── cui_classifer.py
├── load_Dataset.py
├── make_config.py
├── setup.py
├── config.yaml
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

Update the dataset paths inside `config.yaml`.

---

## Workflow

### Step 1 — Dataset Preparation

Configure the ROCOv2 dataset location inside `config.yaml`.

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
* Full-scale GPU training pending

---

## Future Improvements

* Concept-guided report generation
* Beam search decoding
* BLEU, ROUGE and CIDEr evaluation
* RadGraph-based evaluation
* Multi-GPU training
* Clinical error analysis

---

## License

This project is intended for research and educational purposes.

The ROCOv2 dataset is distributed under its respective license. Please refer to the original dataset documentation for usage terms.

---

## Author
Tejashwini Malge




