"""
Data loading for evaluation datasets.
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple


def load_eval_dataset(path: str, max_examples: Optional[int] = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load evaluation dataset from JSONL file.
    
    Args:
        path: Path to JSONL file
        max_examples: Maximum number of examples to load (None = all)
        
    Returns:
        Tuple of (examples_list, dataset_hash)
    """
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

