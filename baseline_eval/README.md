# Baseline Evaluation Framework

This directory contains the baseline evaluation framework for measuring the performance of the untuned foundation model (`microsoft/Phi-3-mini-4k-instruct`) before fine-tuning.

## Overview

The baseline evaluation measures how well the base model follows policies when those policies are explicitly provided in the prompt. This establishes a fair baseline for comparing post-SFT model performance.

## What Gets Measured

### 1. Policy Decision Accuracy (Primary Metric)
- Overall accuracy: `predicted_decision == ground_truth_decision`
- Accuracy per `issue_type` (returns, fraud, warranty, etc.)
- Accuracy per `region` (US, CA, EU, UK, AU)
- Accuracy per `customer_tier` (Guest, Member, VIP, Employee)

### 2. Policy Compliance Errors
- **Violations**: Hard policy failures (e.g., refund approved outside return window)
- **Ambiguous**: Cases where compliance cannot be determined
- **Compliant**: Policy-following responses

### 3. JSON / Schema Validity
- Valid JSON response rate
- Missing required fields
- Invalid data types

### 4. Operational Metrics
- Average latency (milliseconds)
- Input/output tokens per request
- Total tokens processed

## Directory Structure

```
baseline_eval/
├── prompts/
│   └── baseline_prompt.txt          # Prompt template with policy injection
├── results/                          # Output directory (created automatically)
│   ├── baseline_metrics.json        # Summary metrics
│   └── baseline_detailed_results.jsonl  # Per-example results
├── policy_checks.py                  # Policy validation logic
├── metrics.py                        # Metrics collection and aggregation
├── run_baseline.py                   # Main evaluation script
├── cache_model.py                    # Pre-cache model script (RunPod)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── RUNNING_GUIDE.md                  # Detailed running instructions
```

## Setup

1. Install dependencies:
```bash
pip install -r baseline_eval/requirements.txt
```

2. Ensure you have a GPU available (recommended) or use CPU (slower):
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

3. (Optional) Pre-cache the model for faster loading:
```bash
# Cache model to persistent storage (RunPod)
python baseline_eval/cache_model.py \
    --device cuda \
    --cache-dir /workspace/hf_cache

# Or use default cache location
python baseline_eval/cache_model.py --device cuda
```

**Note**: For detailed instructions on running locally vs on RunPod, see [RUNNING_GUIDE.md](RUNNING_GUIDE.md).

## Usage

### Step 1: Split the Dataset (First Time Only)

Before running baseline evaluation, split the 10k dataset:
```bash
# This creates:
# - data/processed/baseline_eval_dataset_300.jsonl (300 examples for baseline)
# - data/processed/sft_training_dataset_9700.jsonl (9700 examples for training)
python split_dataset.py
```

### Step 2: Run Baseline Evaluation

**Full baseline evaluation (300 examples) - RECOMMENDED:**
```bash
python baseline_eval/run_baseline.py \
    --dataset data/processed/baseline_eval_dataset_300.jsonl \
    --max-examples 300 \
    --output-dir baseline_eval/results
```

**Quick test (preview dataset - 100 examples):**
```bash
python baseline_eval/run_baseline.py \
    --dataset data/processed/sft_training_dataset_preview_100.jsonl \
    --max-examples 100 \
    --output-dir baseline_eval/results
```

### Custom Configuration

```bash
python baseline_eval/run_baseline.py \
    --dataset data/processed/sft_training_dataset_10000.jsonl \
    --model-name microsoft/Phi-3-mini-4k-instruct \
    --max-examples 300 \
    --device cuda \
    --max-new-tokens 512 \
    --temperature 0.1 \
    --output-dir baseline_eval/results
```

### RunPod with Persistent Storage

```bash
# 1. Split dataset (first time only)
python split_dataset.py

# 2. Cache model (first time only)
python baseline_eval/cache_model.py --device cuda --cache-dir /workspace/hf_cache

# 3. Run baseline evaluation
python baseline_eval/run_baseline.py \
    --dataset data/processed/baseline_eval_dataset_300.jsonl \
    --max-examples 300 \
    --device cuda \
    --cache-dir /workspace/hf_cache \
    --output-dir baseline_eval/results
```

### Arguments

- `--dataset`: Path to evaluation dataset (JSONL format)
- `--model-name`: Hugging Face model identifier (default: `microsoft/Phi-3-mini-4k-instruct`)
- `--max-examples`: Maximum number of examples to evaluate (default: 300)
- `--baseline-version`: Baseline version identifier (default: `baseline_v1`)
- `--output-dir`: Directory to save results (default: `baseline_eval/results`)
- `--device`: Device to use (`auto`, `cuda`, or `cpu`)
- `--max-new-tokens`: Maximum tokens to generate (default: 512)
- `--temperature`: Generation temperature (default: 0.1, lower = more deterministic)
- `--cache-dir`: Cache directory for models (e.g., `/workspace/hf_cache` for RunPod persistent storage)
- `--use-4bit`: Use 4-bit quantization (QLoRA-compatible). Default: False (full precision for baseline)

## Output

### Summary Metrics (`baseline_metrics.json`)

```json
{
  "baseline_info": {
    "baseline_version": "baseline_v1",
    "base_model": "microsoft/Phi-3-mini-4k-instruct",
    "policy_version": "policy_v1",
    "total_examples": 300,
    "eval_set_hash": "a1b2c3d4e5f6g7h8"
  },
  "decision_accuracy": {
    "overall_accuracy": 71.4,
    "by_issue_type": {...},
    "by_region": {...},
    "by_customer_tier": {...}
  },
  "policy_compliance": {
    "violation_rate": 12.3,
    "violations": 37,
    "compliant": 250
  },
  "json_validity": {
    "valid_json_rate": 96.8,
    "missing_fields": {...},
    "invalid_types": {...}
  },
  "operational_metrics": {
    "avg_latency_seconds": 1.9,
    "avg_input_tokens": 4200,
    "avg_output_tokens": 150
  }
}
```

### Detailed Results (`baseline_detailed_results.jsonl`)

Each line contains per-example results:
```json
{
  "example_idx": 0,
  "issue_type": "return_exchange",
  "region": "US",
  "tier": "Guest",
  "ground_truth_decision": "Eligible",
  "predicted_decision": "Eligible",
  "is_correct": true,
  "is_valid_json": true,
  "latency_ms": 1850.5,
  "input_tokens": 4200,
  "output_tokens": 145
}
```

## Baseline Lock-In

After running baseline evaluation, **freeze these artifacts**:

1. **Baseline Version**: `baseline_v1` (or your version identifier)
2. **Base Model**: `microsoft/Phi-3-mini-4k-instruct`
3. **Policy Version**: `policy_v1`
4. **Eval Set**: The exact dataset used (hash is automatically computed and stored)
5. **Baseline Metrics**: `baseline_metrics.json`

### Eval Set Immutability

**Critical Rule**: Never reuse training samples in eval set.

The framework enforces this by:
- Computing a SHA256 hash of the eval set
- Storing the hash in `baseline_metrics.json`
- Requiring the same hash for all future comparisons

**Do NOT**:
- Regenerate the eval set
- Mix training and eval data
- Modify the eval set after baseline is locked

From now on, every QLoRA model will be compared to this baseline. No changes allowed without bumping version.

## Policy Validation

The `policy_checks.py` module implements policy validation logic that:

- Checks return/exchange eligibility rules
- Validates shipping delay compensation thresholds
- Ensures fraud claims require identity verification
- Verifies warranty period compliance
- Checks payment issue handling
- Validates order modification rules
- Enforces promotion/pricing restrictions

This same validation logic will be reused for drift detection after SFT.

## Expected Baseline Performance

Based on typical foundation model behavior with policy-in-prompt:

- **Decision Accuracy**: 60-75% (varies by issue type)
- **JSON Validity**: 85-95% (model may include extra text)
- **Policy Violations**: 10-20% (model may not strictly follow all rules)
- **Latency**: 1-3 seconds per request (depends on hardware)

After SFT, you should see:
- Accuracy increases (target: 85-95%)
- Violations decrease (target: <5%)
- Tokens decrease significantly (no policy in prompt needed)
- Latency decreases (shorter prompts)

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--max-new-tokens`
- Use CPU instead of CUDA: `--device cpu`
- Process fewer examples: `--max-examples 100`

### JSON Parsing Errors
- The model may include markdown code blocks or extra text
- The `extract_json_from_response()` function handles common cases
- Check `baseline_detailed_results.jsonl` for parse errors

### Slow Inference
- Use GPU: `--device cuda`
- Reduce `--max-new-tokens`
- Use a smaller model variant if available

## Next Steps

### Immediate Action: Run and Lock Baseline

**Next step = RUN IT and lock the baseline**

1. **Split the dataset** (first time only):
   ```bash
   python split_dataset.py
   ```

2. **Run baseline evaluation on 300 examples**:
   ```bash
   python baseline_eval/run_baseline.py \
       --dataset data/processed/baseline_eval_dataset_300.jsonl \
       --max-examples 300 \
       --baseline-version baseline_v1
   ```

2. **Review metrics** (expect "meh" performance — that's normal for untuned model)

3. **Save and freeze**:
   - `baseline_metrics.json` (contains baseline_version and eval_set_hash)
   - `baseline_detailed_results.jsonl`
   - Freeze as `baseline_v1`

### After Baseline is Locked

1. Review `baseline_metrics.json` to understand current performance
2. Identify issue types with lowest accuracy (target for SFT)
3. Note policy violation patterns
4. Proceed to QLoRA fine-tuning
5. Re-evaluate post-SFT and compare to baseline

**Important**: Do NOT proceed to QLoRA until baseline is locked and metrics are saved.

