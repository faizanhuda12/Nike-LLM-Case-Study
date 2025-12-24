# Inference Module

Modular inference system for fine-tuned QLoRA models. Matches the training module structure for consistency.

## Structure

```
inference/
├── __init__.py          # Package initialization
├── utils.py            # Utility functions (seed setting, JSON extraction)
├── prompt_builder.py   # Prompt building with chat templates
├── model_loader.py     # Model and adapter loading
├── inference_runner.py # Inference execution
├── data_loader.py      # Dataset loading
└── README.md           # This file
```

## Components

### `utils.py`
- `set_seed()`: Set random seeds for reproducibility
- `extract_json_from_response()`: Robust JSON extraction from model responses

### `prompt_builder.py`
- `build_user_message()`: Build user message from example
- `build_prompt()`: Build prompt using model's chat template (matches training format)

### `model_loader.py`
- `load_finetuned_model()`: Load base model and apply LoRA adapters
  - Handles 4-bit quantization
  - Disables gradient checkpointing for inference
  - Enforces max length consistency

### `inference_runner.py`
- `run_inference()`: Execute inference on a single prompt
  - Handles tokenization
  - Manages generation parameters
  - Returns metrics (latency, tokens, truncation)

### `data_loader.py`
- `load_eval_dataset()`: Load evaluation dataset from JSONL
  - Computes dataset hash for reproducibility

## Usage

The main entry point is `inference_finetuned.py` which orchestrates all components:

```bash
python inference_finetuned.py \
    --adapter-path models/qlora_sft_qwen25_v1 \
    --dataset baseline_eval/data/baseline_eval_dataset_300.jsonl \
    ...
```

## Key Features

- **Format Consistency**: Uses chat templates matching training format
- **Deterministic**: Default temperature=0.0 for reproducible evaluation
- **Optimized**: Gradient checkpointing disabled, KV cache enabled
- **Robust**: Handles edge cases (truncation, JSON parsing, errors)
- **Modular**: Easy to test and extend individual components

## Design Principles

1. **Matches Training Format**: Uses same chat template and prompt structure
2. **Reproducible**: Deterministic generation by default
3. **Enterprise-Ready**: Modular, testable, maintainable
4. **Compatible**: Uses same metrics system as baseline evaluation

