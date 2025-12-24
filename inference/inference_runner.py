"""
Inference execution for single prompts.
Handles tokenization, generation, and response decoding.
"""

import time
from typing import Tuple

import torch


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
    use_kv_cache: bool = True,
    max_length: int = 2048,
) -> Tuple[str, float, int, int, bool]:
    """
    Run inference on a single prompt.
    
    For evaluation: use temperature=0.0 and do_sample=False for deterministic results.
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = deterministic)
        device: Device to run on
        use_kv_cache: Whether to use KV cache
        max_length: Maximum input sequence length
        
    Returns:
        Tuple of (generated_text, latency_ms, input_tokens, output_tokens, prompt_truncated)
    """
    start_time = time.time()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_tokens = inputs.input_ids.shape[1]
    prompt_truncated = input_tokens >= max_length

    # Configure sampling behavior: deterministic when temperature is 0.0
    do_sample = temperature > 0.0
    
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=use_kv_cache,
            )
        except (AttributeError, TypeError) as e:
            # Handle cache compatibility issues with fallback to no cache
            # Some model configurations may not support KV cache
            if "seen_tokens" in str(e) or "DynamicCache" in str(e):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
            else:
                raise

    # Handle different output formats
    if isinstance(outputs, tuple):
        output_ids = outputs[0]
    else:
        output_ids = outputs
    
    if output_ids.numel() == 0:
        raise ValueError("Model generated empty output")
    
    if len(output_ids.shape) == 2:
        generated_ids = output_ids[0]
    elif len(output_ids.shape) == 1:
        generated_ids = output_ids
    else:
        raise ValueError(f"Unexpected output shape: {output_ids.shape}")
    
    if len(generated_ids) <= input_tokens:
        raise ValueError(f"Generated sequence too short: {len(generated_ids)} <= {input_tokens}")
    
    generated = tokenizer.decode(
        generated_ids[input_tokens:],
        skip_special_tokens=True
    )

    output_tokens = len(generated_ids) - input_tokens
    latency_ms = (time.time() - start_time) * 1000

    return generated, latency_ms, input_tokens, output_tokens, prompt_truncated

