"""
Baseline evaluation script for untuned foundation model.
Runs inference with policy in prompt and collects comprehensive metrics.
"""

import json
import argparse
import time
import re
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from metrics import BaselineMetrics


# -----------------------
# Determinism
# -----------------------
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def load_prompt_template() -> str:
    template_path = Path(__file__).parent / "prompts" / "baseline_prompt.txt"
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(example: Dict[str, Any], template: str) -> str:
    instruction = example.get("instruction", "")
    input_data = example.get("input", {})
    
    # Leak detection: ensure output is never included in the prompt
    assert "output" not in input_data, "Leak detected: output present in model input"
    
    # Ensure output is never included in the prompt (only used for evaluation)
    # Create a clean copy of input_data without any output field
    clean_input = {k: v for k, v in input_data.items() if k != "output"}
    
    policy_document = clean_input.get("policy_document", "")
    input_json = json.dumps(clean_input, indent=2, ensure_ascii=False)

    prompt = template.replace("{{policy_document}}", policy_document)
    prompt = prompt.replace("{{instruction}}", instruction)
    prompt = prompt.replace("{{input}}", input_json)

    return prompt


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return None


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
    use_kv_cache: bool = False
) -> Tuple[str, float, int, int, bool]:

    start_time = time.time()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)

    input_tokens = inputs.input_ids.shape[1]

    # Prompt truncation visibility
    prompt_truncated = input_tokens >= 2048

    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=use_kv_cache,  # Enable for faster generation
            )
        except (AttributeError, TypeError) as e:
            # Fallback: Some models may have cache compatibility issues
            if "seen_tokens" in str(e) or "DynamicCache" in str(e):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,  # Fallback to no cache
                )
            else:
                raise

    # Handle different output formats - model.generate() returns a tensor
    # Shape is typically [batch_size, sequence_length] or [sequence_length]
    if isinstance(outputs, tuple):
        output_ids = outputs[0]
    else:
        output_ids = outputs
    
    # Ensure we have a valid tensor
    if output_ids.numel() == 0:
        raise ValueError(f"Model generated empty output")
    
    # Get the first sequence (batch index 0) if batch dimension exists
    if len(output_ids.shape) == 2:
        generated_ids = output_ids[0]  # [batch, seq] -> [seq]
    elif len(output_ids.shape) == 1:
        generated_ids = output_ids  # Already [seq]
    else:
        raise ValueError(f"Unexpected output shape: {output_ids.shape}")
    
    # Ensure we have enough tokens
    if len(generated_ids) <= input_tokens:
        raise ValueError(f"Generated sequence too short: {len(generated_ids)} <= {input_tokens}")
    
    generated = tokenizer.decode(
        generated_ids[input_tokens:],
        skip_special_tokens=True
    )

    output_tokens = len(generated_ids) - input_tokens
    latency_ms = (time.time() - start_time) * 1000

    return generated, latency_ms, input_tokens, output_tokens, prompt_truncated


def load_eval_dataset(path: str, max_examples: Optional[int]):
    examples = []
    hash_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            if line.strip():
                examples.append(json.loads(line))
                hash_lines.append(line.strip())

    eval_hash = hashlib.sha256(
        "\n".join(hash_lines).encode("utf-8")
    ).hexdigest()[:16]

    return examples, eval_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/processed/sft_training_dataset_preview_100.jsonl",
        help="Path to evaluation dataset (JSONL)"
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--output-dir", default="baseline_eval/results")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--baseline-version", default="baseline_v1")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--use-kv-cache", action="store_true", help="Use KV cache for faster generation (recommended for Qwen)")

    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("Only batch_size=1 is supported for baseline evaluation")

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
        os.makedirs(args.cache_dir, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    if args.use_4bit:
        from transformers import BitsAndBytesConfig

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )

    model_kwargs = {
        "trust_remote_code": True,
        "cache_dir": args.cache_dir
    }

    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
        model_kwargs["device_map"] = "auto" if device == "cuda" else None

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    prompt_template = load_prompt_template()
    examples, eval_hash = load_eval_dataset(args.dataset, args.max_examples)

    print(f"\nLoaded {len(examples)} examples for evaluation")
    print(f"Eval set hash: {eval_hash}")
    print(f"Starting evaluation...\n")

    metrics = BaselineMetrics(
        baseline_version=args.baseline_version,
        eval_set_hash=eval_hash
    )

    for i, example in enumerate(examples):
        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            print(f"Progress: {i + 1}/{len(examples)} examples processed", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
        prompt = build_prompt(example, prompt_template)

        try:
            response, latency, in_tok, out_tok, truncated = run_inference(
                model, tokenizer, prompt,
                args.max_new_tokens,
                args.temperature,
                device,
                use_kv_cache=args.use_kv_cache
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

        metrics.collect(
            example=example,
            prediction=prediction,
            raw_response=response,
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            parse_error=parse_error
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary metrics
    metrics_file = output_dir / "baseline_metrics.json"
    metrics.save(str(metrics_file))
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Save detailed results
    detailed_file = output_dir / "baseline_detailed_results.jsonl"
    with open(detailed_file, "w", encoding="utf-8") as f:
        for result in metrics.example_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Detailed results saved to: {detailed_file}")
    
    # Print summary
    metrics.print_summary()


if __name__ == "__main__":
    main()
