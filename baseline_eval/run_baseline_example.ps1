# Example PowerShell command to run baseline evaluation

# First, split the dataset (run once):
# python split_dataset.py

# Full baseline evaluation (300 examples) - recommended
python baseline_eval/run_baseline.py `
    --dataset data/processed/baseline_eval_dataset_300.jsonl `
    --max-examples 300 `
    --output-dir baseline_eval/results

# RunPod: Use cached model from persistent storage
# python baseline_eval/run_baseline.py `
#     --dataset data/processed/baseline_eval_dataset_300.jsonl `
#     --max-examples 300 `
#     --device cuda `
#     --cache-dir /workspace/hf_cache `
#     --output-dir baseline_eval/results

# RunPod: Use 4-bit quantization (QLoRA-compatible)
# python baseline_eval/run_baseline.py `
#     --dataset data/processed/baseline_eval_dataset_300.jsonl `
#     --max-examples 300 `
#     --device cuda `
#     --cache-dir /workspace/hf_cache `
#     --use-4bit `
#     --output-dir baseline_eval/results

# Quick test (preview dataset - 100 examples)
# python baseline_eval/run_baseline.py `
#     --dataset data/processed/sft_training_dataset_preview_100.jsonl `
#     --max-examples 100 `
#     --output-dir baseline_eval/results

