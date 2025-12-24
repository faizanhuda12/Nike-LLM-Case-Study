"""
Training configuration classes and validation
"""

from dataclasses import dataclass, asdict
from typing import List, Optional
import json
from pathlib import Path


@dataclass
class RunConfig:
    """Complete training run configuration for reproducibility"""
    model_name: str
    dataset_path: str
    output_dir: str
    cache_dir: Optional[str]
    max_seq_length: int
    val_split: float
    seed: int

    # QLoRA
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]

    # Train
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    optim: str
    lr_scheduler_type: str
    max_grad_norm: float

    # Precision
    bf16: bool
    fp16: bool

    # Data
    packing: bool

    def save(self, output_dir: Path) -> None:
        """Save configuration to JSON file"""
        config_path = output_dir / "run_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

