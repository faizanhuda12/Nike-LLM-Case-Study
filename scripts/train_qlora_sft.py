#!/usr/bin/env python3
"""
QLoRA + SFT (Instruction Fine-Tuning) for E-commerce Policy Compliance
Target model: Qwen/Qwen2.5-1.5B-Instruct (lightweight + strong)

Enterprise-grade fine-tuning script with modular architecture.
Uses QLoRA (4-bit quantization + LoRA adapters) for efficient training.

Expected dataset JSONL format:
{
  "instruction": "...",
  "input": {... "policy_document": "...", ...},
  "output": {"decision": "...", "reason": "...", ...}
}

Run example:
python train_qlora_sft.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_path data/processed/sft_training_dataset_3000.jsonl \
  --output_dir models/qlora_sft_qwen25_v1 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --max_seq_length 2048 \
  --cache_dir /workspace/hf_cache
"""

import sys
import argparse
import logging
import traceback
from pathlib import Path

from training.config import RunConfig
from training.utils import (
    setup_logging,
    set_seed,
    setup_cache,
    pick_precision,
    validate_args,
    log_system_info,
)
from training.data_loader import build_datasets
from training.model_setup import load_model_and_tokenizer, default_qwen_target_modules
from training.trainer import train_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    p = argparse.ArgumentParser(
        description="QLoRA + SFT Training for E-commerce Policy Compliance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Base model to fine-tune")
    p.add_argument("--dataset_path", type=str, required=True,
                   help="Path to training dataset (JSONL format)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for trained model and artifacts")

    # Optional paths
    p.add_argument("--cache_dir", type=str, default=None,
                   help="HuggingFace cache directory (e.g., /workspace/hf_cache)")

    # Data
    p.add_argument("--max_seq_length", type=int, default=2048,
                   help="Maximum sequence length for training")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Validation split ratio (0.0 to 1.0)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16,
                   help="LoRA rank (higher = more parameters, more capacity)")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha (scaling factor, typically 2x lora_r)")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout rate")
    p.add_argument("--target_modules", type=str, default="auto",
                   help="Target modules for LoRA: 'auto' or comma-separated list")

    # Training
    p.add_argument("--num_train_epochs", type=float, default=3,
                   help="Number of training epochs")
    p.add_argument("--per_device_train_batch_size", type=int, default=4,
                   help="Batch size per device (adjust for GPU memory)")
    p.add_argument("--per_device_eval_batch_size", type=int, default=4,
                   help="Evaluation batch size per device")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--learning_rate", type=float, default=2e-4,
                   help="Learning rate")
    p.add_argument("--warmup_ratio", type=float, default=0.03,
                   help="Warmup ratio")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="Weight decay")
    p.add_argument("--logging_steps", type=int, default=10,
                   help="Logging frequency")
    p.add_argument("--save_steps", type=int, default=500,
                   help="Checkpoint save frequency")
    p.add_argument("--eval_steps", type=int, default=500,
                   help="Evaluation frequency")
    p.add_argument("--save_total_limit", type=int, default=3,
                   help="Maximum number of checkpoints to keep")
    p.add_argument("--optim", type=str, default="paged_adamw_8bit",
                   help="Optimizer (paged_adamw_8bit recommended for QLoRA)")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine",
                   help="Learning rate scheduler type")
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Maximum gradient norm for clipping")

    # Data behavior
    p.add_argument("--packing", action="store_true",
                   help="Pack multiple short samples into one sequence for speed")

    return p.parse_args()


def main() -> None:
    """Main training function with enterprise-grade error handling"""
    try:
        args = parse_args()
        validate_args(args)
        
        # Setup output directory and logging early
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(output_dir)
        
        log_system_info(logger)
        
        logger.info("=" * 72)
        logger.info("QLoRA + SFT Training - Enterprise Production Run")
        logger.info("=" * 72)
        
        set_seed(args.seed)
        setup_cache(args.cache_dir)

        # Target modules
        if args.target_modules == "auto":
            target_modules = default_qwen_target_modules()
            logger.info(f"Using auto-detected Qwen target modules: {target_modules}")
        else:
            target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
            logger.info(f"Using custom target modules: {target_modules}")

        bf16, fp16 = pick_precision()
        logger.info(f"Precision: bf16={bf16}, fp16={fp16}")

        # Build run configuration
        run_cfg = RunConfig(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            max_seq_length=args.max_seq_length,
            val_split=args.val_split,
            seed=args.seed,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            optim=args.optim,
            lr_scheduler_type=args.lr_scheduler_type,
            max_grad_norm=args.max_grad_norm,
            bf16=bf16,
            fp16=fp16,
            packing=args.packing,
        )

        # Save run config early
        run_cfg.save(output_dir)
        logger.info(f"Saved run configuration to: {output_dir / 'run_config.json'}")

        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            logger=logger,
        )

        # Load and format dataset
        logger.info("Loading and formatting dataset...")
        train_ds, val_ds = build_datasets(
            args.dataset_path, 
            tokenizer, 
            args.val_split, 
            args.seed,
            logger=logger
        )
        logger.info(f"Dataset ready: Train={len(train_ds)}, Val={len(val_ds) if val_ds else 0}")

        # Execute training
        train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=run_cfg,
            output_dir=output_dir,
            logger=logger,
        )

        logger.info("Next steps:")
        logger.info("1) Run baseline_eval/run_baseline.py using this fine-tuned adapter")
        logger.info("2) Compare decision accuracy + policy violations vs baseline_v1")
        logger.info("3) Wire retraining into CI/CD only AFTER you have stable eval gates")
        
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.error("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Training failed with error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
