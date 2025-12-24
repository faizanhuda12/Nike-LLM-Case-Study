# Running Baseline Evaluation: Local vs RunPod

This guide explains how to run the baseline evaluation script locally or on RunPod (GPU cloud platform).

## Quick Decision Guide

**Run Locally If:**
- You have a GPU (NVIDIA) with 8GB+ VRAM
- You want to test quickly before committing to Azure
- You're doing initial development/debugging
- You have good internet (model download ~2-3GB)

**Run on RunPod If:**
- You don't have a local GPU
- You want fast GPU inference
- You need cost-effective GPU access
- You want to avoid local resource constraints

---

## Option 1: Run Locally (Recommended for First Run)

### Prerequisites

1. **Python 3.8+** installed
2. **CUDA-capable GPU** (recommended) or CPU (slower)
3. **Internet connection** (for model download)

### Setup Steps

1. **Install dependencies:**
   ```bash
   pip install -r baseline_eval/requirements.txt
   ```

2. **Verify GPU availability (if using GPU):**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Split the dataset (First time only):**
   ```bash
   # Split 10k dataset into 300 for eval and 9700 for training
   python split_dataset.py
   ```

4. **Run the evaluation:**
   ```bash
   # Quick test (preview dataset - 100 examples, ~5-10 minutes)
   python baseline_eval/run_baseline.py \
       --dataset data/processed/sft_training_dataset_preview_100.jsonl \
       --max-examples 100 \
       --baseline-version baseline_v1

   # Full baseline (300 examples, ~15-30 minutes) - RECOMMENDED
   python baseline_eval/run_baseline.py \
       --dataset data/processed/baseline_eval_dataset_300.jsonl \
       --max-examples 300 \
       --baseline-version baseline_v1 \
       --device cuda  # or 'cpu' if no GPU
   ```

### Expected Performance

- **With GPU (CUDA)**: ~1-3 seconds per example
- **With CPU**: ~5-15 seconds per example
- **Total time for 300 examples**: 
  - GPU: ~5-15 minutes
  - CPU: ~25-75 minutes

### Troubleshooting Local Execution

**Out of Memory (OOM):**
```bash
# Use CPU instead
--device cpu

# Or reduce max tokens
--max-new-tokens 256
```

**Model Download Issues:**
- First run downloads ~2-3GB model from Hugging Face
- Ensure stable internet connection
- Model is cached in `~/.cache/huggingface/` after first download

---

## Option 2: Run on RunPod (Recommended for GPU)

Best for: Fast GPU inference without local GPU

1. **Create a RunPod Pod:**
   - Go to https://www.runpod.io
   - Create a new Pod with GPU (e.g., RTX 3090, RTX 4090, or A100)
   - Choose PyTorch template or custom environment
   - Wait for pod to start (~1-2 minutes)

2. **Connect to Pod:**
   - Use Jupyter Lab or SSH connection
   - Or use RunPod's web terminal

3. **Upload your code and data:**
   - Upload the entire project directory (or clone from your repo)
   - Ensure `data/processed/sft_training_dataset_10000.jsonl` is available

4. **Split the dataset (First time only):**
   ```bash
   # Split 10k dataset into 300 for eval and 9700 for training
   python split_dataset.py
   ```
   
   This creates:
   - `data/processed/baseline_eval_dataset_300.jsonl` (for baseline evaluation)
   - `data/processed/sft_training_dataset_9700.jsonl` (for QLoRA training)

5. **Install dependencies:**
   ```bash
   pip install -r baseline_eval/requirements.txt
   ```

6. **Load and cache the model (Recommended):**
   
   Pre-download the model to avoid delays during evaluation:
   ```bash
   # Cache to persistent storage (recommended for RunPod):
   python baseline_eval/cache_model.py \
       --device cuda \
       --cache-dir /workspace/hf_cache
   
   # Or use default cache location (~/.cache/huggingface/):
   python baseline_eval/cache_model.py --device cuda
   ```
   
   This will:
   - Download the model (~2-3GB) on first run
   - Cache it to disk in 4-bit format (QLoRA-compatible)
   - Test that inference works correctly
   
   **Note:** The model is automatically cached on first use, but pre-caching ensures it's ready before your evaluation run.

7. **Run the baseline evaluation:**
   ```bash
   # Using the cached model from persistent storage:
   python baseline_eval/run_baseline.py \
       --dataset data/processed/baseline_eval_dataset_300.jsonl \
       --max-examples 300 \
       --baseline-version baseline_v1 \
       --device cuda \
       --cache-dir /workspace/hf_cache \
       --output-dir baseline_eval/results
   
   # Or without specifying cache (uses default location):
   python baseline_eval/run_baseline.py \
       --dataset data/processed/baseline_eval_dataset_300.jsonl \
       --max-examples 300 \
       --baseline-version baseline_v1 \
       --device cuda \
       --output-dir baseline_eval/results
   ```
   
   **Note:** 
   - By default, baseline evaluation uses full precision (no quantization)
   - If you want to use 4-bit quantization (matching the cached model), add `--use-4bit` flag
   - The eval dataset (300 examples) is fixed and immutable - use the same set for all baseline comparisons

**Cost**: ~$0.20-1.50/hour for GPU pods (pay per second when running)
**Performance**: Much faster than CPU - expect ~5-10 minutes for 300 examples

---

## Recommended Approach

### For Your First Baseline Run:

1. **Start Local** (if you have GPU):
   - Fastest iteration
   - Easy debugging
   - No cloud setup needed

2. **If Local Fails or No GPU**:
   - Use RunPod Pod
   - Fast GPU access
   - Pay per second billing
   - Similar to local development

3. **For Production Runs**:
   - Use RunPod with persistent storage
   - Save results to cloud storage
   - Can run multiple evaluations in parallel

---

## Next Steps After Running

1. **Review results:**
   - Check `baseline_eval/results/baseline_metrics.json`
   - Review violation patterns

2. **Lock the baseline:**
   - Save `baseline_metrics.json` 
   - Note the `eval_set_hash`
   - Document as `baseline_v1`

3. **Proceed to QLoRA fine-tuning:**
   - Use the same eval set for post-SFT comparison
   - Compare metrics to baseline

---

## Cost Estimates

**Local:**
- Free (your hardware/electricity)

**Azure ML Compute Instance:**
- GPU (NC6s_v3): ~$0.90/hour
- GPU (NC12s_v3): ~$1.80/hour
- For 300 examples (~15 min): ~$0.23-0.45

**Azure ML Job:**
- Same as compute instance
- Pay only for execution time

---

## Troubleshooting

### Local Issues

**"CUDA out of memory":**
- Use `--device cpu`
- Or reduce `--max-new-tokens`

**"Model download failed":**
- Check internet connection
- Try: `huggingface-cli login` (if needed)
- Model cached after first download

### RunPod Issues

**"Pod not starting":**
- Check GPU availability in your region
- Try a different GPU type
- Ensure you have sufficient credits

**"CUDA not available":**
- Verify GPU is attached to pod
- Check PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Restart pod if needed

**"Out of memory":**
- Use a larger GPU (e.g., A100 instead of RTX 3090)
- Or reduce `--max-new-tokens`

**"Model not cached / downloading every time":**
- Check cache location: `python -c "from transformers.utils import TRANSFORMERS_CACHE; print(TRANSFORMERS_CACHE)"`
- Default cache: `~/.cache/huggingface/`
- For persistent storage, use `--cache-dir` flag when running `cache_model.py`
- Verify cache exists: `ls -lh ~/.cache/huggingface/hub/`

**"Model download slow or failing":**
- RunPod pods have good internet, but first download can take 5-10 minutes
- Use `cache_model.py` to pre-download before evaluation
- Check disk space: `df -h` (need ~5GB free for model + cache)

