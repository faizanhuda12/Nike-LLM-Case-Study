# Training Module

Enterprise-grade QLoRA + SFT training module for fine-tuning language models on e-commerce policy compliance tasks.

## Module Structure

```
training/
├── __init__.py          # Package initialization
├── config.py            # Configuration classes (RunConfig)
├── utils.py             # Utilities (logging, validation, reproducibility)
├── data_loader.py       # Dataset loading and formatting
├── model_setup.py       # Model and tokenizer setup with QLoRA
├── trainer.py           # Training orchestration
└── README.md           # This file
```

## Components

### `config.py`
- `RunConfig`: Complete training configuration dataclass
- Configuration validation and serialization

### `utils.py`
- `setup_logging()`: Structured logging setup
- `set_seed()`: Reproducibility utilities
- `setup_cache()`: HuggingFace cache configuration
- `pick_precision()`: Automatic precision selection (bf16/fp16)
- `validate_args()`: Command-line argument validation
- `log_system_info()`: System information logging

### `data_loader.py`
- `load_jsonl()`: Load and validate JSONL datasets
- `build_user_message()`: Format user messages from examples
- `build_assistant_message()`: Format assistant responses
- `format_as_chat()`: Apply chat template formatting
- `build_datasets()`: Create train/val datasets

### `model_setup.py`
- `default_qwen_target_modules()`: Qwen-specific LoRA target modules
- `load_model_and_tokenizer()`: Load model with QLoRA configuration

### `trainer.py`
- `create_training_arguments()`: Build TrainingArguments from config
- `train_model()`: Execute training and save results

## Usage

The main training script (`train_qlora_sft.py`) imports and uses these modules:

```python
from training.config import RunConfig
from training.utils import setup_logging, set_seed, ...
from training.data_loader import build_datasets
from training.model_setup import load_model_and_tokenizer
from training.trainer import train_model
```

## Design Principles

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Reusability**: Components can be imported and used independently
3. **Testability**: Each module can be tested in isolation
4. **Enterprise-ready**: Production-grade error handling and logging
5. **Reproducibility**: Complete configuration tracking and seed management

## Extending

To add new features:
1. Add functionality to the appropriate module
2. Update `RunConfig` if new configuration is needed
3. Update the main script to use new features
4. Maintain backward compatibility where possible

