"""
Model loading for fine-tuned QLoRA models.
Loads base model and applies LoRA adapters.
"""

import os
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    cache_dir: Optional[str] = None,
    use_4bit: bool = True,
    max_length: int = 2048,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load base model and apply LoRA adapters.
    
    Args:
        base_model_name: Base model name (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
        adapter_path: Path to LoRA adapter directory
        cache_dir: HuggingFace cache directory
        use_4bit: Whether to use 4-bit quantization
        max_length: Maximum sequence length (enforced for consistency)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading base model: {base_model_name}")
    
    # Set cache directories if provided
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    if use_4bit:
        print("Loading base model in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
        )
    else:
        print("Loading base model in full precision...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    
    base_model.eval()
    
    # Load LoRA adapters
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # CRITICAL: Disable gradient checkpointing for inference (faster, lower latency)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True
    
    # Force max length consistency
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = max_length
    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        # Don't override if model's max is smaller
        if model.config.max_position_embeddings >= max_length:
            model.config.max_position_embeddings = max_length
    
    print("Model loaded successfully!")
    print(f"Max length: {max_length} (tokenizer: {getattr(tokenizer, 'model_max_length', 'N/A')}, model: {getattr(model.config, 'max_position_embeddings', 'N/A') if hasattr(model, 'config') else 'N/A'})")
    return model, tokenizer

