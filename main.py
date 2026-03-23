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
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, DefaultDataCollator, default_data_collator
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from bert_score import score, BERTScorer
from sacrebleu import corpus_bleu
from data_classes import ParaNMTIterableDataset, CachedIterableDataset, build_cache, build_paranmt_multi_cache, CachedParaNMTMultiDataset, _safe_torch_load


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

def generate_predictions(model, loader, tokenizer, max_length=128):
    model.eval()
    model_for_gen = model.module if hasattr(model, "module") else model
    preds, refs = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            generated_ids = model_for_gen.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_length,
                num_beams=4   # important
            )

            pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            ref_text = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            preds.extend(pred_text)
            refs.extend(ref_text)

    return preds, refs

def evaluate(model, loader, tokenizer, eval_mode="gather_preds"):
    preds, refs = generate_predictions(model, loader, tokenizer)

    # Gather preds/refs across ranks (if DDP)
    if is_dist_avail_and_initialized():
        gathered_preds = _gather_objects(preds)
        gathered_refs = _gather_objects(refs)
        # Flatten
        preds = [p for sub in gathered_preds for p in sub]
        refs = [r for sub in gathered_refs for r in sub]

    if not is_main_process():
        return None

    try:
        bert = compute_bertscore(preds, refs)
    except Exception as e:
        if is_main_process():
            logger.exception("BERTScore failed; setting to 0.0", exc_info=e)
        bert = 0.0
    bleu = compute_bleu(preds, refs)

    return {
        "bertscore": bert,
        "bleu": bleu,
        # "diversity": diversity
    }

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

    r = 2
    target_modules = ['q']
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
    if is_main_process():
        logger.info("After applying LoRA")
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        logger.info("%s total parameters.", f"{total_params:,}")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        logger.info("%s training parameters.", f"{total_trainable_params:,}")

    #####################################
    # LOAD DATASET
    #####################################
    # dataset = ParaNMTIterableDataset(
    #     datapath,
    #     tokenizer=tokenizer,
    #     tokenize_on_the_fly=True
    # )

    # loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=32,
    #     num_workers=4,
    #     pin_memory=True
    # )
    #####################################
    # BUILD CACHE
    #####################################
    # build_cache(
    #     input_path=datapath,
    #     output_dir="dataset/cache_paranmt",
    #     tokenizer=tokenizer,
    #     tokenize=True   # 🔥 switch off if you want raw cache
    # )
    # dataset = CachedIterableDataset(
    #     "dataset/cache_paranmt",
    #     tokenized=True
    # )
    #####################################
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
    dataset = CachedParaNMTMultiDataset(
        cache_dir,
        tokenized=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_data_collator,
        persistent_workers=True,
        drop_last=True
    )

    # Ensure all ranks run the same number of training steps
    steps_per_epoch = int(os.environ.get("STEPS_PER_EPOCH", "0"))
    if steps_per_epoch <= 0 and use_ddp:
        steps_per_epoch = compute_min_steps_per_epoch(cache_dir, batch_size=32, num_workers=4)
        if is_main_process():
            logger.info(f"Using steps_per_epoch={steps_per_epoch} (min across ranks)")
    elif steps_per_epoch > 0 and is_main_process():
        logger.info(f"Using steps_per_epoch={steps_per_epoch} (from env)")

    #####################################
    #####################################
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
    dataset = CachedParaNMTMultiDataset(
        cache_dir,
        tokenized=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=default_data_collator,
        persistent_workers=True
    )

    #####################################
    #####################################
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
    dataset = CachedParaNMTMultiDataset(
        cache_dir,
        tokenized=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
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
                if use_ddp:
                    torch.save(model.module.state_dict(), "best_model.pt")
                else:
                    torch.save(model.state_dict(), "best_model.pt")
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
