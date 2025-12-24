#!/usr/bin/env python3
"""
Compare a single prompt on both baseline and fine-tuned models.
Shows side-by-side responses for quick testing.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Import baseline metrics for prompt building
baseline_eval_path = Path(__file__).parent / "baseline_eval"
sys.path.insert(0, str(baseline_eval_path))

# Import inference modules
from inference.utils import set_seed, extract_json_from_response
from inference.prompt_builder import build_prompt


def load_baseline_model(model_name: str, cache_dir: str = None, use_4bit: bool = True):
    """Load baseline model (no adapters)"""
    print(f"Loading baseline model: {model_name}")
    
    if cache_dir:
        import os
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True
    
    return model, tokenizer


def load_finetuned_model(base_model_name: str, adapter_path: str, cache_dir: str = None, use_4bit: bool = True):
    """Load fine-tuned model (base + adapters)"""
    from inference.model_loader import load_finetuned_model as load_ft
    return load_ft(base_model_name, adapter_path, cache_dir, use_4bit, max_length=2048)


def run_single_inference(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0):
    """Run inference on a single prompt"""
    from inference.inference_runner import run_inference
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    response, latency, in_tok, out_tok, truncated = run_inference(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
        use_kv_cache=True,
        max_length=2048,
    )
    
    return response, latency, in_tok, out_tok, truncated


def format_response(response: str, max_length: int = 500) -> str:
    """Format response for display"""
    if len(response) > max_length:
        return response[:max_length] + "... [truncated]"
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Compare a single prompt on baseline vs fine-tuned model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--prompt-file",
        help="Path to JSON file with example (must have 'instruction' and 'input' keys)"
    )
    
    parser.add_argument(
        "--instruction",
        help="Instruction text (if not using --prompt-file)"
    )
    
    parser.add_argument(
        "--input-json",
        help="Input JSON as string (if not using --prompt-file)"
    )
    
    parser.add_argument(
        "--base-model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name"
    )
    
    parser.add_argument(
        "--adapter-path",
        default="models/qlora_sft_qwen25_v1",
        help="Path to fine-tuned LoRA adapter directory"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="/workspace/hf_cache",
        help="HuggingFace cache directory"
    )
    
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = deterministic)"
    )
    
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load example
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            example = json.load(f)
    elif args.instruction and args.input_json:
        example = {
            "instruction": args.instruction,
            "input": json.loads(args.input_json)
        }
    else:
        print("Error: Must provide either --prompt-file or both --instruction and --input-json")
        return
    
    print("=" * 80)
    print("SINGLE PROMPT COMPARISON: Baseline vs Fine-Tuned")
    print("=" * 80)
    print()
    
    # Load models
    print("Loading models...")
    baseline_model, baseline_tokenizer = load_baseline_model(
        args.base_model_name,
        args.cache_dir,
        args.use_4bit
    )
    print("Baseline model loaded")
    
    finetuned_model, finetuned_tokenizer = load_finetuned_model(
        args.base_model_name,
        args.adapter_path,
        args.cache_dir,
        args.use_4bit
    )
    print("Fine-tuned model loaded")
    print()
    
    # Build prompts
    baseline_prompt = build_prompt(example, baseline_tokenizer)
    finetuned_prompt = build_prompt(example, finetuned_tokenizer)
    
    print("=" * 80)
    print("PROMPT:")
    print("=" * 80)
    print(f"Instruction: {example.get('instruction', 'N/A')}")
    print(f"Input keys: {list(example.get('input', {}).keys())}")
    if 'output' in example:
        print(f"Expected output: {json.dumps(example['output'], indent=2)}")
    print()
    
    # Run inference
    print("Running inference...")
    print()
    
    print("Baseline model:")
    baseline_response, baseline_latency, baseline_in_tok, baseline_out_tok, baseline_trunc = run_single_inference(
        baseline_model, baseline_tokenizer, baseline_prompt,
        args.max_new_tokens, args.temperature
    )
    
    print("Fine-tuned model:")
    finetuned_response, finetuned_latency, finetuned_in_tok, finetuned_out_tok, finetuned_trunc = run_single_inference(
        finetuned_model, finetuned_tokenizer, finetuned_prompt,
        args.max_new_tokens, args.temperature
    )
    
    print()
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    # Baseline response
    print("BASELINE MODEL:")
    print("-" * 80)
    print(f"Response: {format_response(baseline_response)}")
    baseline_json = extract_json_from_response(baseline_response)
    if baseline_json:
        print(f"Parsed JSON: {json.dumps(baseline_json, indent=2)}")
    else:
        print("Warning: Could not parse JSON from response")
    print(f"Latency: {baseline_latency:.2f}ms | Tokens: {baseline_in_tok}→{baseline_out_tok}")
    print()
    
    # Fine-tuned response
    print("FINE-TUNED MODEL:")
    print("-" * 80)
    print(f"Response: {format_response(finetuned_response)}")
    finetuned_json = extract_json_from_response(finetuned_response)
    if finetuned_json:
        print(f"Parsed JSON: {json.dumps(finetuned_json, indent=2)}")
    else:
        print("Warning: Could not parse JSON from response")
    print(f"Latency: {finetuned_latency:.2f}ms | Tokens: {finetuned_in_tok}→{finetuned_out_tok}")
    print()
    
    # Comparison
    print("=" * 80)
    print("COMPARISON:")
    print("=" * 80)
    if 'output' in example:
        expected = example['output']
        baseline_match = baseline_json == expected if baseline_json else False
        finetuned_match = finetuned_json == expected if finetuned_json else False
        
        print(f"Expected: {json.dumps(expected, indent=2)}")
        print(f"Baseline matches: {'Yes' if baseline_match else 'No'}")
        print(f"Fine-tuned matches: {'Yes' if finetuned_match else 'No'}")
    else:
        print("No expected output provided for comparison")
    
    print(f"Latency difference: {finetuned_latency - baseline_latency:+.2f}ms")
    print()


if __name__ == "__main__":
    main()

