"""
Model and tokenizer setup with QLoRA configuration
"""

import torch
import logging
from typing import Any, List, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


def default_qwen_target_modules() -> List[str]:
    """
    Good default for Qwen2.5 (decoder-only transformer):
    attention: q_proj, k_proj, v_proj, o_proj
    mlp: gate_proj, up_proj, down_proj
    """
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model_and_tokenizer(
    model_name: str,
    cache_dir: Optional[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[torch.nn.Module, Any]:
    """Load model and tokenizer with QLoRA setup and validation"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be very slow on CPU.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    logger.info(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    # Make sure we have padding token (important for batching / eval)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token")

    logger.info("Configuring 4-bit QLoRA (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    logger.info(f"Loading model in 4-bit: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            device_map="auto",
            quantization_config=bnb_config,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Critical for stable k-bit training
    logger.info("Preparing model for k-bit training with gradient checkpointing...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA
    logger.info(f"Configuring LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {target_modules}")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    logger.info("Trainable parameters:")
    model.print_trainable_parameters()

    return model, tokenizer

