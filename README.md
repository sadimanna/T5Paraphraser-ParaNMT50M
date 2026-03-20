# Paraphraser

This directory contains a small training and data-prep workflow for paraphrase modeling on ParaNMT-style data.

**What’s here**
- `main.py`: training script using Hugging Face Transformers + PEFT LoRA. It builds cached datasets, creates dataloaders, and sets up model/tokenizer.
- `data_classes.py`: dataset utilities, cache builders, and iterable dataset implementations.
- `dataset/`: local data and scripts.
- `train.py`, `utils.py`: currently empty placeholders.

**Key components**
- Cache builders:
  - `build_cache(...)`: simple tokenized cache for a single paraphrase pair per line.
  - `build_paranmt_multi_cache(...)`: builds multi‑paraphrase groups using SBERT similarity, saves shards as `.pt`.
- Dataset classes:
  - `ParaNMTIterableDataset`: streams raw text and tokenizes on the fly.
  - `CachedIterableDataset`: streams cached shards.
  - `ParaNMTMultiDataset`: builds multi‑paraphrase groups on the fly.
  - `CachedParaNMTMultiDataset`: streams cached multi‑paraphrase shards.

**Data layout**
- `dataset/para-nmt-50m/` and `dataset/para-nmt-5m-processed/` hold raw/processed ParaNMT data.
- `dataset/split_paranmt.py` splits `para-nmt-5m-processed.txt` into train/val/test.
- `dataset/*_cache_multi/` contains cached shards produced by `build_paranmt_multi_cache(...)`.

**Quickstart**
1. Split the ParaNMT data:
```bash
python paraphraser/dataset/split_paranmt.py
```
2. Run the training script:
```bash
python paraphraser/main.py
```

**Important notes**
- `main.py` currently uses absolute paths like `/dataset/para-nmt-5m-processed/train.txt`. Adjust these to your local paths.
- `main.py` imports `Trainer` from Transformers, which can trigger a TensorFlow/Keras import. If you don’t need TF, run:
```bash
TRANSFORMERS_NO_TF=1 python paraphraser/main.py
```
or install `tf-keras` to satisfy the dependency.

