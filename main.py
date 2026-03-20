import torch
import logging
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from data_classes import ParaNMTIterableDataset, CachedIterableDataset, build_cache, build_paranmt_multi_cache, CachedParaNMTMultiDataset
import argparse
import os
import glob
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType

from bert_score import score

def compute_bertscore(preds, refs):
    P, R, F1 = score(preds, refs, lang="en", verbose=False)
    return F1.mean().item()

from sacrebleu import corpus_bleu

def compute_bleu(preds, refs):
    return corpus_bleu(preds, [refs]).score

def generate_predictions(model, loader, tokenizer, max_length=128):
    model.eval()
    preds, refs = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}

            generated_ids = model.generate(
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

def evaluate(model, loader, tokenizer):
    preds, refs = generate_predictions(model, loader, tokenizer)

    bert = compute_bertscore(preds, refs)
    bleu = compute_bleu(preds, refs)
    # diversity = distinct_n(preds)

    return {
        "bertscore": bert,
        "bleu": bleu,
        # "diversity": diversity
    }

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger.info("Starting paraphraser training script")
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
    model.to("cuda")
    model.eval()
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
    # print(model)
    logger.info("After applying LoRA")
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("%s total parameters.", f"{total_params:,}")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
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
    logger.info("Building train cache")
    build_paranmt_multi_cache(
        input_path=train_datapath,
        output_dir="dataset/train_cache_multi",
        tokenizer=tokenizer,
        tokenize=True,   # or False for flexible experiments
        group_size=5
    )
    dataset = CachedParaNMTMultiDataset(
        "dataset/train_cache_multi",
        tokenized=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    #####################################
    logger.info("Building val cache")
    build_paranmt_multi_cache(
        input_path=train_datapath,
        output_dir="dataset/val_cache_multi",
        tokenizer=tokenizer,
        tokenize=True,   # or False for flexible experiments
        group_size=5
    )
    dataset = CachedParaNMTMultiDataset(
        "dataset/val_cache_multi",
        tokenized=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    #####################################
    logger.info("Building test cache")
    build_paranmt_multi_cache(
        input_path=train_datapath,
        output_dir="dataset/test_cache_multi",
        tokenizer=tokenizer,
        tokenize=True,   # or False for flexible experiments
        group_size=5
    )
    dataset = CachedParaNMTMultiDataset(
        "dataset/test_cache_multi",
        tokenized=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    #####################################
    # TRAINING LOOP
    #####################################
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=3e-4)

    model.train()

    best_val_score = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

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

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # =========================
        # Validation
        # =========================
        val_metrics = evaluate(model, val_loader, tokenizer)

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val BERTScore: {val_metrics['bertscore']:.4f}")
        print(f"Val BLEU: {val_metrics['bleu']:.2f}")
        print(f"Val Diversity: {val_metrics['diversity']:.4f}")

        # =========================
        # Model selection
        # =========================
        if val_metrics["bertscore"] > best_val_score:
            best_val_score = val_metrics["bertscore"]
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model!")

    ######################################
    # EVALUATION
    ######################################
    model.eval()

    model.load_state_dict(torch.load("best_model.pt"))

    test_metrics = evaluate(model, test_loader, tokenizer)

    print("\nTest Results:")
    print(test_metrics)
