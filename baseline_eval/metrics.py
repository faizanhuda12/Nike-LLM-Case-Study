"""
Metrics collection for baseline evaluation.
Tracks decision accuracy, policy compliance, JSON validity,
language compliance, and operational metrics.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional, List

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from policy_checks import check_all_policies


class BaselineMetrics:
    """Collects and aggregates baseline evaluation metrics."""

    def __init__(self, baseline_version: str = "baseline_v1", eval_set_hash: str = None):
        self.baseline_version = baseline_version
        self.eval_set_hash = eval_set_hash
        self.reset()

    def reset(self):
        # ---------------------------
        # Counters
        # ---------------------------
        self.total_examples = 0
        self.decision_attempts = 0
        self.correct_decisions = 0

        # Accuracy slices
        self.accuracy_by_issue_type = defaultdict(lambda: {"correct": 0, "total": 0})
        self.accuracy_by_region = defaultdict(lambda: {"correct": 0, "total": 0})
        self.accuracy_by_tier = defaultdict(lambda: {"correct": 0, "total": 0})

        # Policy compliance
        self.policy_violations = 0
        self.policy_compliant = 0
        self.policy_ambiguous = 0
        self.violation_details = []

        # JSON / schema validity
        self.valid_json_count = 0
        self.invalid_json_count = 0
        self.missing_fields = defaultdict(int)
        self.invalid_types = defaultdict(int)

        # Language compliance
        self.language_violations = 0

        # Operational metrics
        self.latencies_ms: List[float] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

        # Per-example logs
        self.example_results = []

    def collect(
        self,
        example: Dict[str, Any],
        prediction: Dict[str, Any],
        raw_response: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        parse_error: Optional[str] = None,
    ):
        self.total_examples += 1
        self.total_requests += 1
        self.latencies_ms.append(latency_ms)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        input_data = example.get("input", {})
        ground_truth = example.get("output", {})

        # ---------------------------
        # JSON validity
        # ---------------------------
        is_valid_json = parse_error is None

        if is_valid_json:
            required_fields = [
                "decision",
                "reason",
                "customer_response",
                "next_action",
                "escalation_required",
            ]
            for field in required_fields:
                if field not in prediction:
                    self.missing_fields[field] += 1
                    is_valid_json = False

            if "escalation_required" in prediction:
                if not isinstance(prediction["escalation_required"], bool):
                    self.invalid_types["escalation_required"] += 1
                    is_valid_json = False

        if is_valid_json:
            self.valid_json_count += 1
        else:
            self.invalid_json_count += 1

        # ---------------------------
        # Language compliance (simple heuristic)
        # ---------------------------
        if is_valid_json:
            response_text = prediction.get("customer_response", "")
            if any(ch in response_text for ch in ["¡", "¿"]):
                self.language_violations += 1

        # ---------------------------
        # Decision accuracy
        # ---------------------------
        is_correct = False
        if is_valid_json:
            self.decision_attempts += 1
            gt_decision = ground_truth.get("decision", "")
            pred_decision = prediction.get("decision", "")

            is_correct = gt_decision == pred_decision
            if is_correct:
                self.correct_decisions += 1

            issue_type = input_data.get("issue_type", "unknown")
            region = input_data.get("region", "unknown")
            tier = input_data.get("customer_tier", "unknown")

            self.accuracy_by_issue_type[issue_type]["total"] += 1
            self.accuracy_by_region[region]["total"] += 1
            self.accuracy_by_tier[tier]["total"] += 1

            if is_correct:
                self.accuracy_by_issue_type[issue_type]["correct"] += 1
                self.accuracy_by_region[region]["correct"] += 1
                self.accuracy_by_tier[tier]["correct"] += 1

            # ---------------------------
            # Policy compliance
            # ---------------------------
            is_compliant, violation_type, details = check_all_policies(
                input_data, prediction
            )

            if not is_compliant:
                self.policy_violations += 1
                self.violation_details.append(
                    {
                        "example_idx": self.total_examples - 1,
                        "issue_type": issue_type,
                        "violation_type": violation_type,
                        "details": details,
                        "predicted_decision": pred_decision,
                        "ground_truth_decision": gt_decision,
                    }
                )
            elif violation_type == "ambiguous":
                self.policy_ambiguous += 1
            else:
                self.policy_compliant += 1
        else:
            self.policy_ambiguous += 1

        # ---------------------------
        # Per-example log
        # ---------------------------
        self.example_results.append(
            {
                "example_idx": self.total_examples - 1,
                "issue_type": input_data.get("issue_type", "unknown"),
                "region": input_data.get("region", "unknown"),
                "tier": input_data.get("customer_tier", "unknown"),
                "is_valid_json": is_valid_json,
                "is_correct": is_correct,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "parse_error": parse_error,
            }
        )

    def get_summary(self) -> Dict[str, Any]:
        if self.total_examples == 0:
            return {"error": "No examples processed"}

        overall_accuracy = (
            (self.correct_decisions / self.decision_attempts) * 100
            if self.decision_attempts > 0
            else 0
        )

        def calc_accuracy(bucket):
            return {
                k: round(v["correct"] / v["total"] * 100, 2)
                for k, v in bucket.items()
                if v["total"] > 0
            }

        latencies = np.array(self.latencies_ms)

        return {
            "baseline_info": {
                "baseline_version": self.baseline_version,
                "base_model": "microsoft/Phi-3-mini-4k-instruct",
                "policy_version": "policy_v1",
                "eval_set_hash": self.eval_set_hash or "unknown",
                "total_examples": self.total_examples,
            },
            "decision_accuracy": {
                "overall_accuracy": round(overall_accuracy, 2),
                "correct_decisions": self.correct_decisions,
                "decision_attempts": self.decision_attempts,
                "by_issue_type": calc_accuracy(self.accuracy_by_issue_type),
                "by_region": calc_accuracy(self.accuracy_by_region),
                "by_customer_tier": calc_accuracy(self.accuracy_by_tier),
            },
            "policy_compliance": {
                "violations": self.policy_violations,
                "compliant": self.policy_compliant,
                "ambiguous": self.policy_ambiguous,
            },
            "json_validity": {
                "valid_json_rate": round(
                    self.valid_json_count / self.total_examples * 100, 2
                ),
                "valid": self.valid_json_count,
                "invalid": self.invalid_json_count,
                "missing_fields": dict(self.missing_fields),
                "invalid_types": dict(self.invalid_types),
            },
            "language_compliance": {
                "language_violations": self.language_violations
            },
            "operational_metrics": {
                "avg_latency_ms": round(latencies.mean(), 2),
                "p50_latency_ms": round(np.percentile(latencies, 50), 2),
                "p95_latency_ms": round(np.percentile(latencies, 95), 2),
                "avg_input_tokens": round(
                    self.total_input_tokens / self.total_requests, 1
                ),
                "avg_output_tokens": round(
                    self.total_output_tokens / self.total_requests, 1
                ),
                "total_tokens": self.total_input_tokens
                + self.total_output_tokens,
            },
            "violation_details": self.violation_details[:20],
        }

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print a human-readable summary of metrics."""
        summary = self.get_summary()
        
        if "error" in summary:
            print(f"Error: {summary['error']}")
            return
        
        print("\n" + "=" * 70)
        print("BASELINE EVALUATION SUMMARY")
        print("=" * 70)
        
        info = summary["baseline_info"]
        print(f"\nBaseline Version: {info['baseline_version']}")
        print(f"Total Examples: {info['total_examples']}")
        print(f"Eval Set Hash: {info['eval_set_hash']}")
        
        acc = summary["decision_accuracy"]
        print(f"\nDecision Accuracy: {acc['overall_accuracy']}%")
        print(f"  Correct: {acc['correct_decisions']}/{acc['decision_attempts']}")
        
        comp = summary["policy_compliance"]
        print(f"\nPolicy Compliance:")
        print(f"  Compliant: {comp['compliant']}")
        print(f"  Violations: {comp['violations']}")
        print(f"  Ambiguous: {comp['ambiguous']}")
        
        json_val = summary["json_validity"]
        print(f"\nJSON Validity: {json_val['valid_json_rate']}%")
        print(f"  Valid: {json_val['valid']}, Invalid: {json_val['invalid']}")
        
        ops = summary["operational_metrics"]
        print(f"\nOperational Metrics:")
        print(f"  Avg Latency: {ops['avg_latency_ms']} ms")
        print(f"  P95 Latency: {ops['p95_latency_ms']} ms")
        print(f"  Avg Input Tokens: {ops['avg_input_tokens']}")
        print(f"  Avg Output Tokens: {ops['avg_output_tokens']}")
        print(f"  Total Tokens: {ops['total_tokens']}")
        
        print("=" * 70 + "\n")