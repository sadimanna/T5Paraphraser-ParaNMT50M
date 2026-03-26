import logging
import psutil
import torch
import argparse
import os, sys
import glob
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration, default_data_collator
# from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from bert_score import score, BERTScorer
from sacrebleu import corpus_bleu
from data_classes import ParaNMTIterableDataset, build_paranmt_multi_cache, CachedParaNMTMultiDataset, _safe_torch_load


def compute_bertscore(preds, refs):
    scorer = _get_bertscorer()
    P, R, F1 = scorer.score(preds, refs)
    return F1.mean().item()

def compute_bleu(preds, refs):
    return corpus_bleu(preds, [refs]).score

_BERT_SCORER = None

def _get_device():
    dev = globals().get("device", None)
    if dev is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dev

def _get_bertscorer():
    global _BERT_SCORER
    if _BERT_SCORER is None:
        _BERT_SCORER = BERTScorer(lang="en", rescale_with_baseline=True, device=_get_device())
        tok = _BERT_SCORER._tokenizer
        if not hasattr(tok, "build_inputs_with_special_tokens"):
            def build_inputs_with_special_tokens(token_ids_0, token_ids_1=None):
                if token_ids_1 is None:
                    return [tok.cls_token_id] + token_ids_0 + [tok.sep_token_id]
                return [tok.cls_token_id] + token_ids_0 + [tok.sep_token_id, tok.sep_token_id] + token_ids_1 + [tok.sep_token_id]
            tok.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    return _BERT_SCORER

def generate_predictions(model, loader, tokenizer, max_length=128, max_eval_steps = 10000):
    model.eval()
    model_for_gen = model.module if hasattr(model, "module") else model
    preds, refs = [], []

    with torch.no_grad():
        loader_iter = iter(loader)
        for step in range(max_eval_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break # Both ranks should ideally hit this at the same time
            if is_main_process() and step % 100 == 0:
                logger.info(f"generate_predictions step {step}")
            batch = {k: v.to(device) for k, v in batch.items()}

            generated_ids = model_for_gen.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_length,
                num_beams=4   # important
            )

            pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = batch["labels"]
            if (labels == -100).any():
                pad_id = tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                labels = labels.masked_fill(labels == -100, pad_id)
            ref_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # print(pred_text)
            # print(ref_text)

            preds.extend(pred_text)
            refs.extend(ref_text)

    return preds, refs

def evaluate(model, loader, tokenizer, eval_mode="gather_preds"):
    preds, refs = generate_predictions(model, loader, tokenizer)

    if eval_mode == "gather_preds":
        # Gather preds/refs across ranks (if DDP)
        if is_dist_avail_and_initialized():
            gathered_preds = _gather_objects(preds)
            gathered_refs = _gather_objects(refs)
            preds = [p for sub in gathered_preds for p in sub]
            refs = [r for sub in gathered_refs for r in sub]

        if not is_main_process():
            return None

        # Drop empty candidates/refs to avoid BERTScore warnings
        pairs = [(p, r) for p, r in zip(preds, refs) if p and p.strip() and r and r.strip()]
        dropped = len(preds) - len(pairs)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} empty pred/ref pairs before BERTScore")
        if not pairs:
            return {"bertscore": 0.0, "bleu": 0.0}
        preds, refs = zip(*pairs)
        preds, refs = list(preds), list(refs)

        try:
            bert = compute_bertscore(preds, refs)
        except Exception:
            logger.exception("BERTScore failed; setting to 0.0")
            bert = 0.0
        bleu = compute_bleu(preds, refs)

        return {"bertscore": bert, "bleu": bleu}

    if eval_mode == "gather_metrics":
        # Drop empty candidates/refs locally to avoid BERTScore warnings
        local_pairs = [(p, r) for p, r in zip(preds, refs) if p and p.strip() and r and r.strip()]
        if local_pairs:
            preds, refs = zip(*local_pairs)
            preds, refs = list(preds), list(refs)
        else:
            preds, refs = [], []

        try:
            local_bert = compute_bertscore(preds, refs) if preds else 0.0
        except Exception:
            if is_main_process():
                logger.exception("BERTScore failed; setting to 0.0")
            local_bert = 0.0
        local_bleu = compute_bleu(preds, refs) if preds else 0.0
        local_count = len(preds)

        if is_dist_avail_and_initialized():
            gathered = _gather_objects({"bertscore": local_bert, "bleu": local_bleu, "count": local_count})
            if not is_main_process():
                return None
            total_count = sum(g["count"] for g in gathered)
            if total_count == 0:
                return {"bertscore": 0.0, "bleu": 0.0}
            bert = sum(g["bertscore"] * g["count"] for g in gathered) / total_count
            bleu = sum(g["bleu"] * g["count"] for g in gathered) / total_count
            return {"bertscore": bert, "bleu": bleu}
        else:
            return {"bertscore": local_bert, "bleu": local_bleu}

    # Fallback: local on main
    if not is_main_process():
        return None
    try:
        bert = compute_bertscore(preds, refs)
    except Exception:
        logger.exception("BERTScore failed; setting to 0.0")
        bert = 0.0
    bleu = compute_bleu(preds, refs)
    return {"bertscore": bert, "bleu": bleu}


logger = logging.getLogger(__name__)

def log_gpu_mem(prefix=""):
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / (1024**3)
        reserv_gb = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"{prefix} GPU mem: allocated={alloc_gb:.2f} GB, reserved={reserv_gb:.2f} GB")

def log_cpu_mem(prefix=""):
    rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    logger.info(f"{prefix} CPU RSS: {rss_gb:.2f} GB")

def log_mem(prefix=""):
    log_cpu_mem(prefix)
    log_gpu_mem(prefix)

def compute_min_steps_per_epoch(cache_dir, batch_size, num_workers):
    # Estimate per-rank batches from cached shard sizes and take global min
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    rank = dist.get_rank() if is_dist_avail_and_initialized() else 0
    world_size = dist.get_world_size() if is_dist_avail_and_initialized() else 1
    global_workers = num_workers * world_size
    start = rank * num_workers
    end = start + num_workers
    local_samples = 0
    for file_idx, file in enumerate(files):
        mod = file_idx % global_workers
        if start <= mod < end:
            data = _safe_torch_load(file)
            local_samples += len(data)
    local_steps = local_samples // batch_size
    if is_dist_avail_and_initialized():
        t = torch.tensor(local_steps, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return int(t.item())
    return int(local_steps)

def compute_stream_steps_per_epoch(path, batch_size, world_size):
    # Count total lines in the dataset file (one sample per line).
    with open(path, "rb") as f:
        total_lines = 0
        buf_size = 1024 * 1024
        while True:
            data = f.read(buf_size)
            if not data:
                break
            total_lines += data.count(b"\n")
    if total_lines == 0:
        return 0
    global_batch = batch_size * world_size
    return max(1, total_lines // global_batch)

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not is_dist_avail_and_initialized()) or dist.get_rank() == 0

def _gather_objects(obj):
    if not is_dist_avail_and_initialized():
        return [obj]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered

def ddp_barrier():
    if is_dist_avail_and_initialized():
        dev = globals().get("device", None)
        if dev is not None and hasattr(dev, "index") and dev.index is not None:
            dist.barrier(device_ids=[dev.index])
        else:
            dist.barrier()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Starting paraphraser training script")
    parser = argparse.ArgumentParser(description="Paraphraser training")
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Enable full fine-tuning (no LoRA adapters).",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank (used only if not full fine-tuning).",
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q", "k", "v"],
        help="Target modules for LoRA (used only if not full fine-tuning).",
    )
    args = parser.parse_args()
    eval_mode = os.environ.get("EVAL_MODE", "gather_preds")
    # DDP init (torchrun will set WORLD_SIZE/LOCAL_RANK)
    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r = args.lora_r
    target_modules = args.target_modules
    model_name = "google-t5/t5-small"
    train_datapath = "./dataset/para-nmt-5m-processed/train.txt"
    val_datapath = "./dataset/para-nmt-5m-processed/val.txt"
    test_datapath = "./dataset/para-nmt-5m-processed/test.txt"
    num_epochs = 100
    #####################################
    # LOAD MODEL
    #####################################
    # Model Declaration
    logger.info("Loading model: %s", model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name) #"google/flan-t5-base")
    model.to(device)
    model.eval()
    model_for_gen = model.module if hasattr(model, "module") else model
    # print(model)
    # print("Before Applying LoRA")
    # # Total parameters and trainable parameters.
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"{total_params:,} total parameters.")
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"{total_trainable_params:,} training parameters.")
    # Declaring Tokenizer
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name) #"google/flan-t5-base")
    if args.full_finetune:
        if is_main_process():
            logger.info("Full fine-tuning enabled (LoRA disabled).")
    else:
        # Declaring LoRA Config
        config = LoraConfig(
            r=r,
            lora_alpha=r * 2,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        # Get PEFT-ed Model
        model = get_peft_model(model, config)
    if use_ddp:
        model = DDP(model, device_ids=[device.index], output_device=device.index)
    # print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        if not args.full_finetune:
            logger.info("After applying LoRA")
            logger.info(f"Percentage of trainable params: {100 * total_trainable_params / total_params:.2f}%")
        else:
            logger.info("Full finetuning")
        logger.info("%s total parameters.", f"{total_params:,}")
        logger.info("%s training parameters.", f"{total_trainable_params:,}")

    #####################################
    # LOAD DATASET (Streaming)
    #####################################
    dataset_mode = os.environ.get("DATASET_MODE", "stream").strip().lower()
    if dataset_mode not in {"stream", "cache"}:
        raise ValueError(f"Unsupported DATASET_MODE={dataset_mode!r}. Use 'stream' or 'cache'.")

    if dataset_mode == "cache":
        cache_dir = "./dataset/train_cache_multi"
        if not (os.path.isdir(cache_dir) and os.listdir(cache_dir)):
            if is_main_process():
                logger.info(f"Building train cache at {cache_dir}")
                build_paranmt_multi_cache(
                    input_path=train_datapath,
                    output_dir=cache_dir,
                    tokenizer=tokenizer,
                    tokenize=True,   # or False for flexible experiments
                    group_size=5
                )
            if use_ddp:
                ddp_barrier()
        else:
            if is_main_process():
                logger.info(f"Train cache exists at {cache_dir}; skipping build.")
            if use_ddp:
                ddp_barrier()
        train_dataset = CachedParaNMTMultiDataset(
            cache_dir,
            tokenized=True
        )
    else:
        train_dataset = ParaNMTIterableDataset(
            train_datapath,
            tokenizer=tokenizer,
            tokenize_on_the_fly=True
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_data_collator,
        persistent_workers=True,
        drop_last=True
    )

    # Ensure all ranks run the same number of training steps
    steps_per_epoch = int(os.environ.get("STEPS_PER_EPOCH", "0"))
    if steps_per_epoch > 0 and is_main_process():
        logger.info(f"Using steps_per_epoch={steps_per_epoch} (from env)")
    elif steps_per_epoch <= 0 and use_ddp and dataset_mode == "cache":
        steps_per_epoch = compute_min_steps_per_epoch(cache_dir, batch_size=32, num_workers=4)
        if is_main_process():
            logger.info(f"Using steps_per_epoch={steps_per_epoch} (min across ranks)")
    elif steps_per_epoch <= 0 and use_ddp and dataset_mode == "stream":
        if is_main_process():
            steps_per_epoch = compute_stream_steps_per_epoch(
                train_datapath, batch_size=32, world_size=dist.get_world_size()
            )
            logger.info(f"Using steps_per_epoch={steps_per_epoch} (stream count)")
        if use_ddp:
            if is_dist_avail_and_initialized():
                t = torch.tensor(steps_per_epoch, device=device)
                dist.broadcast(t, src=0)
                steps_per_epoch = int(t.item())
    elif steps_per_epoch <= 0 and use_ddp and is_main_process():
        logger.warning("DDP is enabled but STEPS_PER_EPOCH is not set; ranks may desync on iterable data.")

    #####################################
    #####################################
    if dataset_mode == "cache":
        cache_dir = "./dataset/val_cache_multi"
        if not (os.path.isdir(cache_dir) and os.listdir(cache_dir)):
            if is_main_process():
                logger.info(f"Building val cache at {cache_dir}")
                build_paranmt_multi_cache(
                    input_path=val_datapath,
                    output_dir=cache_dir,
                    tokenizer=tokenizer,
                    tokenize=True,   # or False for flexible experiments
                    group_size=5
                )
            if use_ddp:
                ddp_barrier()
        else:
            if is_main_process():
                logger.info(f"Val cache exists at {cache_dir}; skipping build.")
            if use_ddp:
                ddp_barrier()
        val_dataset = CachedParaNMTMultiDataset(
            cache_dir,
            tokenized=True
        )
    else:
        val_dataset = ParaNMTIterableDataset(
            val_datapath,
            tokenizer=tokenizer,
            tokenize_on_the_fly=True
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_data_collator,
        persistent_workers=True
    )

    #####################################
    #####################################
    if dataset_mode == "cache":
        cache_dir = "./dataset/test_cache_multi"
        if not (os.path.isdir(cache_dir) and os.listdir(cache_dir)):
            if is_main_process():
                logger.info(f"Building test cache at {cache_dir}")
                build_paranmt_multi_cache(
                    input_path=test_datapath,
                    output_dir=cache_dir,
                    tokenizer=tokenizer,
                    tokenize=True,   # or False for flexible experiments
                    group_size=5
                )
            if use_ddp:
                ddp_barrier()
        else:
            if is_main_process():
                logger.info(f"Test cache exists at {cache_dir}; skipping build.")
            if use_ddp:
                ddp_barrier()
        test_dataset = CachedParaNMTMultiDataset(
            cache_dir,
            tokenized=True
        )
    else:
        test_dataset = ParaNMTIterableDataset(
            test_datapath,
            tokenizer=tokenizer,
            tokenize_on_the_fly=True
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_data_collator,
        persistent_workers=True
    )

    #####################################
    # TRAINING LOOP
    #####################################
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=3e-4)

    model.train()

    best_val_score = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_steps = 0
        step = 0

        if steps_per_epoch:
            train_iter = iter(train_loader)
            for _ in range(steps_per_epoch):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                step += 1
                if is_main_process() and step % 1000 == 0:
                    log_mem(prefix=f"epoch {epoch} step {step}")
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )

                loss = outputs.loss

                if "score" in batch:
                    weights = batch["score"].to(device)
                    loss = (loss * weights).mean()

                if is_main_process() and step % 100 == 0:
                    logger.info(f"epoch {epoch} step {step} loss: {loss.item():.4f}")

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()
                num_steps += 1
        else:
            for batch in train_loader:
                step += 1
                if is_main_process() and step % 1000 == 0:
                    log_mem(prefix=f"epoch {epoch} step {step}")
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )

                loss = outputs.loss

                if "score" in batch:
                    weights = batch["score"].to(device)
                    loss = (loss * weights).mean()

                if is_main_process() and step % 100 == 0:
                    logger.info(f"epoch {epoch} step {step} loss: {loss.item():.4f}")

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()
                num_steps += 1
            step += 1
            if is_main_process() and step % 1000 == 0:
                log_mem(prefix=f"epoch {epoch} step {step}")
            batch = {k: v.to(device) for k, v in batch.items()}
            # logger.info(batch["input_ids"])

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            loss = outputs.loss

            # 🔥 score-weighted training (if available)
            if "score" in batch:
                weights = batch["score"].to(device)
                loss = (loss * weights).mean()

            if is_main_process() and step % 100 == 0:
                logger.info(f"epoch {epoch} step {step} loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

        avg_train_loss = total_loss / max(1, num_steps)

        # Epoch memory summary
        if is_main_process():
            log_mem(prefix=f"epoch {epoch} summary")
            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
                logger.info(f"epoch {epoch} peak GPU allocated: {peak_gb:.2f} GB")
                torch.cuda.reset_peak_memory_stats()

        # =========================
        # Validation
        # =========================
        if use_ddp:
            ddp_barrier()
        val_metrics = evaluate(model, val_loader, tokenizer, eval_mode=eval_mode)
        if use_ddp:
            ddp_barrier()

        if is_main_process() and val_metrics is not None:
            print(f"\nEpoch {epoch}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val BERTScore: {val_metrics['bertscore']:.4f}")
            print(f"Val BLEU: {val_metrics['bleu']:.2f}")
            # print(f"Val Diversity: {val_metrics['diversity']:.4f}")

            # =========================
            # Model selection
            # =========================
            if val_metrics["bertscore"] > best_val_score:
                best_val_score = val_metrics["bertscore"]
                if not os.path.exists("./saved_models"):
                    os.makedirs("./saved_models")
                if use_ddp:
                    torch.save(model.module.state_dict(), "saved_models/best_model.pt")
                else:
                    torch.save(model.state_dict(), "saved_models/best_model.pt")
                print("Saved best model!")

    ######################################
    # EVALUATION
    ######################################
    model.eval()
    model_for_gen = model.module if hasattr(model, "module") else model

    # Load best checkpoint on all ranks if using DDP
    if use_ddp:
        if is_main_process():
            state = torch.load("best_model.pt")
            model.module.load_state_dict(state)
        ddp_barrier()
        if not is_main_process():
            state = torch.load("best_model.pt")
            model.module.load_state_dict(state)
    else:
        model.load_state_dict(torch.load("best_model.pt"))

    if use_ddp:
        ddp_barrier()
    test_metrics = evaluate(model, test_loader, tokenizer, eval_mode=eval_mode)
    if use_ddp:
        ddp_barrier()

    if is_main_process() and test_metrics is not None:
        print("\nTest Results:")
        print(test_metrics)

    if use_ddp:
        dist.destroy_process_group()
