"""
Training orchestration and execution module.

Provides version-safe compatibility with TRL SFTTrainer across different library versions.
Implements dynamic parameter detection to handle API changes gracefully.
"""

import json
import logging
import inspect
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from trl import SFTTrainer

from training.config import RunConfig


# ---------------------------------------------------------------------
# TrainingArguments (version-safe)
# ---------------------------------------------------------------------
def create_training_arguments(
    config: RunConfig,
    output_dir: Path,
    val_dataset: Optional[Dataset],
) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    eval_key = "evaluation_strategy" if "evaluation_strategy" in params else "eval_strategy"

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "save_total_limit": config.save_total_limit,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": config.optim,
        "max_grad_norm": config.max_grad_norm,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "report_to": ["none"],
        "save_strategy": "steps",
        "do_eval": val_dataset is not None,
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "load_best_model_at_end": val_dataset is not None,
        "metric_for_best_model": "eval_loss" if val_dataset is not None else None,
        "greater_is_better": False,
    }

    kwargs[eval_key] = "steps" if val_dataset is not None else "no"

    return TrainingArguments(**kwargs)


# ---------------------------------------------------------------------
# Tokenization (always safe)
# ---------------------------------------------------------------------
def tokenize_dataset(dataset: Dataset, tokenizer, max_seq_length: int) -> Dataset:
    if "text" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'text' column")

    def _tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    return dataset.map(
        _tok,
        batched=True,
        remove_columns=dataset.column_names,
    )


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: RunConfig,
    output_dir: Path,
    logger: logging.Logger,
) -> None:

    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        effective_batch *= torch.cuda.device_count()
    logger.info(f"Effective batch size: {effective_batch}")

    training_args = create_training_arguments(config, output_dir, val_dataset)

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_dataset, tokenizer, config.max_seq_length)
    if val_dataset is not None:
        val_dataset = tokenize_dataset(val_dataset, tokenizer, config.max_seq_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # -----------------------------------------------------------------
    # Build SFTTrainer kwargs strictly from its signature
    # -----------------------------------------------------------------
    sig = inspect.signature(SFTTrainer.__init__)
    allowed = sig.parameters

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
    }

    if "data_collator" in allowed:
        kwargs["data_collator"] = data_collator

    # Note: Tokenizer parameter is not passed to SFTTrainer
    # Pre-tokenized datasets are used instead for compatibility across TRL versions

    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(**kwargs)

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("Starting QLoRA + SFT training")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(
        f"Batch: {config.per_device_train_batch_size} x "
        f"{config.gradient_accumulation_steps} = {effective_batch}"
    )
    logger.info("=" * 72)

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    duration_hours = (end_time - start_time).total_seconds() / 3600
    logger.info(f"Training completed in {duration_hours:.2f} hours")

    # Save adapters + tokenizer
    logger.info("Saving LoRA adapters and tokenizer...")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Summary
    summary = {
        "training_completed": True,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_hours": duration_hours,
        "total_steps": trainer.state.global_step,
        "final_train_loss": (
            trainer.state.log_history[-1].get("train_loss")
            if trainer.state.log_history
            else None
        ),
        "best_eval_loss": (
            min(
                [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h],
                default=None,
            )
            if val_dataset is not None
            else None
        ),
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved training summary to: {summary_path}")
    logger.info("=" * 72)
    logger.info("Training completed successfully")
    logger.info("=" * 72)
