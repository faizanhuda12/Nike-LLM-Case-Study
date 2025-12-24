"""
Utility functions for inference
"""

import re
import json
from typing import Dict, Any, Optional

import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from model response using multiple strategies.
    Robust parsing that handles code blocks, raw JSON, and edge cases.
    """
    # Try to find JSON in code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object directly
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try parsing the whole response
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return None

