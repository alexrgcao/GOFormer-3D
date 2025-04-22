# GOFormer-3D
GOFormer‑3D: A Structure‑Aware Transformer for Protein Function Annotation

A PyTorch–based framework for predicting protein functions via Gene Ontology (GO) terms, leveraging ESM‑2 embeddings and AlphaFold2 structural features.

---

## Table of Contents

1. [Requirements](#requirements)  
2. [Project Structure](#project-structure)  
3. [Installation](#installation)  
4. [Data Preparation & Preprocessing](#data-preparation--preprocessing)  
5. [Training](#training)  
6. [Evaluation](#evaluation)  
7. [Configuration](#configuration)

---

## Requirements

- Python 3.8+

Install dependencies with:

```bash
pip install -r requirement.txt
```

Contents of `requirement.txt`:

```
torch
numpy
tqdm
biopython
obonet
accelerate
scikit-learn
Transformers
cafaeval
```

---

## Project Structure

```
.
├── src/
│   ├── train.py                   # Training loop & checkpointing
│   ├── model.py                   # Transformer model definition
│   ├── dataset.py                 # PyTorch Dataset & collate_fn
│   ├── preprocess_panda3d.py      # ESM‑2 & AlphaFold2 feature fetchers
│   ├── download_pdb.py            # UniProt→PDB downloader
│   ├── clean_panda3d_data.py      # Utility to clear heavy fields in PANDA‑3D PKLs
│   ├── labels.py                  # GO‑term vocabulary builder
│   ├── global_atoms_order.py      # Build global atom ordering
│   ├── ground_truth.py            # CAFA5 ground‑truth sampler
│   └── evaluate.py                # Inference & mapping to UniProt
├── requirement.txt                # Python dependencies
└── README.md                      # This file
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/protein-go-prediction.git
cd protein-go-prediction
```

2. Install dependencies:

```bash
pip install -r requirement.txt
```

---

## Data Preparation & Preprocessing

### 1. Obtain PANDA‑3D Dataset

Download and place `.pkl` files under:

```
PANDA3D_ROOT/
```

### 2. Clean PANDA‑3D Pickles

Strip large fields to speed up downstream steps:

```bash
python src/clean_panda3d_data.py \
  --input_root  PANDA3D_ROOT \
  --output_root cleaned_panda3d
```

### 3. Build GO‑Term Vocabulary

```bash
python src/labels.py \
  --input_root                 cleaned_panda3d \
  --vocab_output_json_path     go_vocab_filtered.json
```

### 4. Extract Global Atom Order

Make sure you have downloaded all necessary PDB files.

```bash
python src/global_atoms_order.py \
  --pdb_root     /path/to/pdbs \
  --output_json  global_atom_order.json
```

### 5. Fetch ESM‑2 and AlphaFold2 Features

```bash
python src/preprocess_panda3d.py \
  --input_dir         cleaned_panda3d \
  --output_dir        processed_pkl \
  --global_atom_order global_atom_order.json \
  --local_pdb_dir     /path/to/pdbs
```

---

## Training

Train your model on the processed `.pkl` data:

```bash
python src/train.py \
  --data_dir        processed_pkl \
  --output_dir      models_and_metrics \
  --epochs          50 \
  --batch_size      16 \
  --lr              1e-4 \
  --use_struct_bias True \
  --go_vocab_file   go_vocab_filtered.json \
  --atom_order_file global_atom_order.json
```

---

## Evaluation

### 1. Obtain CAFA 5 Target List

Prepare a FASTA file of target proteins:

```
cafa5_targets.fasta
```

### 2. Generate Ground‑Truth Labels (if not available)

```bash
python src/ground_truth.py \
  --test_ids cafa5_targets.fasta \
  --output   cafa5_ground_truth \
  --subset   0.1 \
  --seed     42
```

### 3. Run Inference

```bash
python src/evaluate.py \
  --model_ckpt   models_and_metrics/model_epoch50.pt \
  --cafa_pkl_dir processed_cafa_pkl \
  --go_vocab     go_vocab_filtered.json \
  --atom_order   global_atom_order.json \
  --output_tsv   cafa5_predictions.tsv \
  --batch_size   8
```

---

## Configuration

All scripts support `--help` to view available arguments.

Common options:
- `--quick_test` – run minimal batches for debugging
- `--max_workers` – number of threads
- `--timeout` – HTTP request timeout in seconds
