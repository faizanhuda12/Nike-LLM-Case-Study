"""
Data loading and formatting for SFT training
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datasets import Dataset


BASE_SYSTEM = (
    "You are an enterprise e-commerce customer operations assistant.\n"
    "You MUST follow the policy document provided below strictly.\n"
    "Do not invent rules. Do not override policy.\n"
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file with validation"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    items: List[Dict[str, Any]] = []
    errors = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Validate required fields
                if "instruction" not in item or "input" not in item or "output" not in item:
                    errors.append(f"Line {line_num}: Missing required fields (instruction, input, output)")
                    continue
                items.append(item)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON decode error: {e}")
    
    if errors:
        logging.warning(f"Found {len(errors)} errors in dataset. First 5: {errors[:5]}")
    
    if len(items) == 0:
        raise ValueError(f"No valid examples found in {path}")
    
    return items


def build_user_message(example: Dict[str, Any]) -> str:
    """Build user message from example"""
    instruction = example.get("instruction", "").strip()
    input_data = example.get("input", {}) or {}
    
    # Prevent data leakage: remove ground-truth output from input data
    if "output" in input_data:
        input_data = {k: v for k, v in input_data.items() if k != "output"}

    policy_document = (input_data.get("policy_document") or "").strip()

    # Keep input JSON readable + stable
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


def build_assistant_message(example: Dict[str, Any]) -> str:
    """Build assistant message (target output) from example"""
    output_data = example.get("output", {}) or {}
    # Generate compact JSON representation to minimize token usage
    return json.dumps(output_data, ensure_ascii=False, separators=(",", ":"))


def format_as_chat(tokenizer, example: Dict[str, Any]) -> str:
    """
    Uses model's chat template when available (Qwen has one).
    Produces a single training text where the assistant output is the target.
    """
    messages = [
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": build_user_message(example)},
        {"role": "assistant", "content": build_assistant_message(example)},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # Fallback format for models without chat template support
    return (
        f"SYSTEM:\n{BASE_SYSTEM}\n\n"
        f"USER:\n{messages[1]['content']}\n\n"
        f"ASSISTANT:\n{messages[2]['content']}\n"
    )


def build_datasets(
    dataset_path: str,
    tokenizer,
    val_split: float,
    seed: int,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Build train/val datasets with validation and logging"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading dataset from: {dataset_path}")
    raw = load_jsonl(dataset_path)
    logger.info(f"Loaded {len(raw)} examples")
    
    if len(raw) < 50 and val_split > 0:
        logger.warning("Very small dataset; consider val_split=0 to avoid unstable eval.")

    # Validate data quality
    logger.info("Formatting examples...")
    texts = []
    format_errors = 0
    for i, ex in enumerate(raw, 1):
        try:
            formatted = format_as_chat(tokenizer, ex)
            texts.append(formatted)
        except Exception as e:
            format_errors += 1
            if format_errors <= 5:
                logger.warning(f"Error formatting example {i}: {e}")
        if i % 1000 == 0:
            logger.info(f"Formatted {i}/{len(raw)} examples...")

    if format_errors > 0:
        logger.warning(f"Skipped {format_errors} examples due to formatting errors")

    if len(texts) == 0:
        raise ValueError("No valid formatted examples. Check dataset format.")

    ds = Dataset.from_dict({"text": texts})

    if val_split and val_split > 0:
        split = ds.train_test_split(test_size=val_split, seed=seed, shuffle=True)
        train_size = len(split["train"])
        val_size = len(split["test"])
        logger.info(f"Dataset split: {train_size} train, {val_size} validation")
        return split["train"], split["test"]
    
    logger.info(f"Using full dataset for training: {len(ds)} examples")
    return ds, None

