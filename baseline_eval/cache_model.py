"""
cache_base_model.py

Pre-download and cache a base LLM for RunPod persistent storage.
This script ensures the model and tokenizer are cached in 4-bit
format exactly as used for QLoRA fine-tuning and evaluation.

Run ONCE per pod to avoid repeated downloads.
"""

import os
import argparse
import torch

# ----------------------------
# Parse arguments FIRST
# ----------------------------
parser = argparse.ArgumentParser(description="Cache base LLM for RunPod")

parser.add_argument(
    "--model-name",
    type=str,
    default="microsoft/Phi-3-mini-4k-instruct",
    help="Hugging Face model ID",
)

parser.add_argument(
    "--cache-dir",
    type=str,
    default="/workspace/hf_cache",
    help="Persistent cache directory (RunPod volume)",
)

parser.add_argument(
    "--device",
    type=str,
    default="auto",
    choices=["auto", "cuda", "cpu"],
    help="Device to use",
)

args = parser.parse_args()

# ----------------------------
# Set HF cache env vars BEFORE imports
# ----------------------------
os.environ["HF_HOME"] = args.cache_dir
os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
os.environ["HF_DATASETS_CACHE"] = args.cache_dir

# ----------------------------
# Imports after env setup
# ----------------------------
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ----------------------------
# Device resolution
# ----------------------------
if args.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args.device

print("=" * 70)
print("Model caching configuration")
print(f"Model:        {args.model_name}")
print(f"Device:       {device}")
print(f"Cache dir:    {args.cache_dir}")
print(f"CUDA enabled: {torch.cuda.is_available()}")
print("=" * 70)
print()

# ----------------------------
# Ensure cache directory exists
# ----------------------------
os.makedirs(args.cache_dir, exist_ok=True)

# ----------------------------
# Load tokenizer (cached)
# ----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    trust_remote_code=True,
    cache_dir=args.cache_dir,
)
print("✓ Tokenizer cached")
print()

# ----------------------------
# 4-bit quantization config (QLoRA-aligned)
# ----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ----------------------------
# Load model (cached, 4-bit)
# ----------------------------
print("Loading model in 4-bit (QLoRA-compatible)...")
print("First run may take a few minutes to download weights.")

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
    cache_dir=args.cache_dir,
)

model.eval()
print("✓ Model cached successfully")
print()

# ----------------------------
# Sanity check inference
# ----------------------------
print("Running sanity-check inference...")

prompt = "Hello! Please respond with a short greeting."
inputs = tokenizer(prompt, return_tensors="pt")

if device == "cuda":
    inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("✓ Inference successful")
print(f"Sample output: {response}")
print()

# ----------------------------
# Final confirmation
# ----------------------------
print("=" * 70)
print("Base model successfully cached and ready.")
print("You may now run baseline evaluation or QLoRA fine-tuning.")
print("=" * 70)
