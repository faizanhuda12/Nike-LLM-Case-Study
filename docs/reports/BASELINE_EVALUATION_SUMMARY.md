# Baseline Evaluation Summary

**Date:** December 23, 2025  
**Model:** Qwen/Qwen2.5-1.5B-Instruct (Untuned/Base Model)  
**Status:** **COMPLETED**

---

## Evaluation Configuration

- **Dataset:** `baseline_eval/data/baseline_eval_dataset_300.jsonl`
- **Total Examples:** 300
- **Eval Set Hash:** `0e0427dcac8e877d`
- **Baseline Version:** `baseline_v1`
- **Device:** CUDA
- **Max New Tokens:** 256
- **KV Cache:** Enabled

---

## Results Summary

### Decision Accuracy
- **Overall Accuracy:** **36.24%**
- **Correct Decisions:** 108 out of 298 (2 examples had parse errors)
- **Incorrect Decisions:** 190

### Policy Compliance
- **Compliant Responses:** 270 (90.0%)
- **Policy Violations:** 28 (9.3%)
- **Ambiguous Cases:** 2 (0.7%)

### JSON Validity
- **Valid JSON:** 298 (99.33%)
- **Invalid JSON:** 2 (0.67%)
- **Parse Errors:** 2

### Operational Performance
- **Average Latency:** 6,218.36 ms (6.22 seconds)
- **P95 Latency:** 6,788.68 ms (6.79 seconds)
- **Average Input Tokens:** 1,183.0
- **Average Output Tokens:** 256.0
- **Total Tokens Processed:** 431,715

---

## Key Observations

### Strengths
**High JSON Validity:** 99.33% - Model produces well-formed JSON consistently  
**Good Policy Compliance:** 90% of responses follow policy guidelines  
**Stable Performance:** Consistent latency across examples

### Areas for Improvement
**Low Decision Accuracy:** 36.24% - Significant room for improvement  
**Policy Violations:** 28 violations (9.3%) - Needs reduction  
**Latency:** 6+ seconds per example - Could be optimized

---

## Baseline Metrics Breakdown

| Metric | Value | Target for Fine-Tuning |
|--------|-------|----------------------|
| Decision Accuracy | 36.24% | >50% |
| Policy Compliance | 90.0% | >95% |
| Policy Violations | 28 | <10 |
| JSON Validity | 99.33% | >99% |
| Avg Latency | 6.22s | <5s |

---

## Output Files

- **Summary Metrics:** `baseline_eval/results/baseline_metrics.json`
- **Detailed Results:** `baseline_eval/results/baseline_detailed_results.jsonl`
- **Evaluation Log:** `baseline_evaluation_log_20251223.log`

---

## Comparison Point

This baseline evaluation establishes the **starting point** for fine-tuning:

- **Before Fine-Tuning:** 36.24% accuracy, 28 policy violations
- **After Fine-Tuning:** Target >50% accuracy, <10 policy violations

The fine-tuned model should demonstrate measurable improvements across all metrics while maintaining the high JSON validity rate.

---

**Note:** This baseline was run before fine-tuning to establish a comparison point. The fine-tuned model evaluation should show improvements in decision accuracy and policy compliance.

