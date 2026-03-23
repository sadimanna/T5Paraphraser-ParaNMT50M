# Paraphraser Training (T5 + LoRA + DDP)

This project trains a paraphrase model using a T5 backbone with LoRA fine‑tuning. Data is cached into shard files to speed up training. The training script supports single‑GPU and multi‑GPU (DDP) runs, and includes optional evaluation aggregation across ranks.

## Project Layout
- `main.py` — training + evaluation loop (T5 + LoRA + DDP)
- `data_classes.py` — dataset builders, cache builders, and iterable cached datasets
- `dataset/` — input data and cached shards
- `run.slurm` — SLURM job script (DDP‑ready)
- `slurm_logs/` — job logs

## How It Works (High Level)
1. **Cache building**: `build_paranmt_multi_cache(...)` tokenizes and groups paraphrases, then writes shard files to `dataset/*_cache_multi`.
2. **Training**: `CachedParaNMTMultiDataset` streams cached shards into a PyTorch `DataLoader`.
3. **LoRA**: `peft` applies low‑rank adapters on the T5 model for efficient fine‑tuning.
4. **DDP**: If launched with `torchrun`, each rank trains on a shard of cached files. Evaluation can either gather predictions or reduce metrics across ranks.

## Requirements
- Python 3.10+
- `torch`, `transformers`, `peft`, `datasets`, `bert_score`, `sacrebleu`, `psutil`

Install (example):
```bash
pip install torch transformers peft datasets bert_score sacrebleu psutil
```

## Running Locally (Single GPU)
```bash
python main.py
```

## Running with DDP (Multi‑GPU, Single Node)
```bash
torchrun --nproc_per_node=2 main.py
```

## Running on SLURM (DDP)
The provided `run.slurm` script is DDP‑ready. Submit it with:
```bash
sbatch run.slurm
```
It uses:
- `#SBATCH --gres=gpu:2`
- `torchrun --nproc_per_node=$NUM_GPUS`

## Evaluation Modes (DDP)
Set `EVAL_MODE` to control how validation/test metrics are aggregated:
- `gather_preds` (default): gather all predictions/refs across ranks, then compute metrics on rank‑0 (most accurate).
- `gather_metrics`: compute metrics per‑rank, then reduce on rank‑0 (approximate for BLEU).

Example:
```bash
EVAL_MODE=gather_preds torchrun --nproc_per_node=2 main.py
```

## Notes
- Cache directories are created in `dataset/train_cache_multi`, `dataset/val_cache_multi`, `dataset/test_cache_multi`.
- Memory logging is enabled every 1000 steps and per epoch; loss is logged every 100 steps.
- If you see NCCL timeouts, make sure all ranks participate in evaluation/collectives.

