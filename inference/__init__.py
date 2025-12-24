"""
Inference module for fine-tuned QLoRA models.
Provides modular components for loading models, building prompts, and running inference.
"""

__version__ = "1.0.0"

# Export main functions for easier imports
from inference.utils import set_seed, extract_json_from_response
from inference.prompt_builder import build_prompt
from inference.model_loader import load_finetuned_model
from inference.inference_runner import run_inference
from inference.data_loader import load_eval_dataset

__all__ = [
    "set_seed",
    "extract_json_from_response",
    "build_prompt",
    "load_finetuned_model",
    "run_inference",
    "load_eval_dataset",
]

