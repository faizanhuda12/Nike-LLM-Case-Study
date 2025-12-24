#!/usr/bin/env python3
"""
Inference script for fine-tuned QLoRA model.
Loads base model + LoRA adapters and runs inference on evaluation dataset.
Compatible with baseline evaluation metrics for comparison.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Import baseline metrics for comparison
baseline_eval_path = Path(__file__).parent / "baseline_eval"
sys.path.insert(0, str(baseline_eval_path))
from metrics import BaselineMetrics

# Import inference modules
from inference.utils import set_seed, extract_json_from_response
from inference.prompt_builder import build_prompt
from inference.model_loader import load_finetuned_model
from inference.inference_runner import run_inference
from inference.data_loader import load_eval_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned QLoRA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--dataset",
        default="baseline_eval/data/baseline_eval_dataset_300.jsonl",
        help="Path to evaluation dataset (JSONL)"
    )
    parser.add_argument(
        "--base-model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name (must match training base model)"
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to fine-tuned LoRA adapter directory"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (None = all)"
    )
    parser.add_argument(
        "--output-dir",
        default="inference_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum input sequence length (must match training)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = deterministic, recommended for eval)"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization for base model"
    )
    parser.add_argument(
        "--use-kv-cache",
        action="store_true",
        help="Use KV cache for faster generation"
    )
    parser.add_argument(
        "--model-version",
        default="sft_v1",
        help="Model version identifier for metrics"
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(42)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model and tokenizer
    model, tokenizer = load_finetuned_model(
        base_model_name=args.base_model_name,
        adapter_path=args.adapter_path,
        cache_dir=args.cache_dir,
        use_4bit=args.use_4bit,
        max_length=args.max_length,
    )

    # Load evaluation dataset
    examples, eval_hash = load_eval_dataset(args.dataset, args.max_examples)

    print(f"\nLoaded {len(examples)} examples for evaluation")
    print(f"Eval set hash: {eval_hash}")
    print(f"Temperature: {args.temperature} ({'deterministic' if args.temperature == 0.0 else 'sampling'})")
    print(f"Max length: {args.max_length}")
    print(f"Starting inference...\n")

    # Initialize metrics
    metrics = BaselineMetrics(
        baseline_version=args.model_version,
        eval_set_hash=eval_hash
    )

    # Run inference on all examples
    for i, example in enumerate(examples):
        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            print(f"Progress: {i + 1}/{len(examples)} examples processed", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
        
        # Build prompt using chat template to ensure format consistency with training
        prompt = build_prompt(example, tokenizer)

        try:
            response, latency, in_tok, out_tok, truncated = run_inference(
                model, tokenizer, prompt,
                args.max_new_tokens,
                args.temperature,
                device,
                use_kv_cache=args.use_kv_cache,
                max_length=args.max_length,
            )

            prediction = extract_json_from_response(response)
            parse_error = None if prediction else "JSON_PARSE_FAILURE"

        except Exception as e:
            response = ""
            latency = 0
            in_tok = 0
            out_tok = 0
            truncated = False
            prediction = {}
            parse_error = str(e)

        # Collect metrics
        metrics.collect(
            example=example,
            prediction=prediction,
            raw_response=response,
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            parse_error=parse_error
        )
        
        # Log truncation events for debugging and quality monitoring
        if truncated:
            print(f"  Warning: Prompt truncated (input_tokens={in_tok} >= max_length={args.max_length})")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary metrics
    metrics_file = output_dir / f"{args.model_version}_metrics.json"
    metrics.save(str(metrics_file))
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Save detailed results
    detailed_file = output_dir / f"{args.model_version}_detailed_results.jsonl"
    with open(detailed_file, "w", encoding="utf-8") as f:
        for result in metrics.example_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Detailed results saved to: {detailed_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 70)
    metrics.print_summary()
    print("\n" + "=" * 70)
    print(f"Compare with baseline: baseline_eval/results/baseline_metrics.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
