"""
Prompt building for inference using chat templates.
Matches training format exactly for consistency.
"""

import json
from typing import Dict, Any


# System prompt matching training format
BASE_SYSTEM = (
    "You are an enterprise e-commerce customer operations assistant.\n"
    "You MUST follow the policy document provided below strictly.\n"
    "Do not invent rules. Do not override policy.\n"
)


def build_user_message(example: Dict[str, Any]) -> str:
    """
    Build user message from example (matches training format).
    
    Args:
        example: Example dict with 'instruction' and 'input' keys
        
    Returns:
        Formatted user message string
    """
    instruction = example.get("instruction", "").strip()
    input_data = example.get("input", {}) or {}
    
    # Prevent data leakage: remove ground-truth output from input data
    if "output" in input_data:
        input_data = {k: v for k, v in input_data.items() if k != "output"}

    policy_document = (input_data.get("policy_document") or "").strip()

    # Serialize input data to JSON with consistent formatting for reproducibility
    input_json = json.dumps(input_data, ensure_ascii=False, separators=(",", ":"), indent=2)

    user_msg = (
        "POLICY DOCUMENT:\n"
        f"{policy_document}\n\n"
        "TASK:\n"
        f"{instruction}\n\n"
        "INPUT:\n"
        f"{input_json}\n\n"
        "OUTPUT:\n"
        "Return ONLY valid JSON with keys:\n"
        "decision, reason, customer_response, next_action, escalation_required.\n"
    )
    return user_msg


def build_prompt(example: Dict[str, Any], tokenizer) -> str:
    """
    Build prompt using model's chat template (CRITICAL: matches training format).
    This ensures format consistency between training and inference.
    
    Args:
        example: Example dict with 'instruction' and 'input' keys
        tokenizer: Tokenizer with chat template support
        
    Returns:
        Formatted prompt string ready for tokenization
    """
    # Data leakage prevention: validate that ground-truth output is not present in input
    input_data = example.get("input", {})
    assert "output" not in input_data, "Data leakage detected: output present in model input"
    
    # Build messages in chat format
    messages = [
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": build_user_message(example)},
    ]
    
    # Use model's chat template (matches training)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # Add assistant prompt for generation
        )
    else:
        # Fallback format for models without chat template support
        # Note: Qwen models should always have chat template support
        prompt = (
            f"SYSTEM:\n{BASE_SYSTEM}\n\n"
            f"USER:\n{messages[1]['content']}\n\n"
            f"ASSISTANT:\n"
        )
    
    return prompt

