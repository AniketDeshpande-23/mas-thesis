"""
Built-in evaluation system for MAS vs Single LLM comparison
Tracks metrics without external dependencies
"""
from collections import defaultdict
from datetime import datetime
import json

# time_efficiency = 1.0 at TIME_BASELINE_S, drops linearly to 0.0 at baseline + scale
TIME_BASELINE_S = 5    # seconds considered "instant" — full efficiency score
TIME_SCALE_S    = 10   # seconds over baseline that maps to 0.0 efficiency


class EvaluationTracker:
    """Lightweight evaluation tracker - no external dependencies"""
    
    def __init__(self, experiment_name: str = "mas_vs_single"):
        self.experiment_name = experiment_name
        self.metrics_log = []
    
    def log_evaluation(self, model_name: str, mode: str, run_num: int, 
                     report_dict: dict, metrics: dict) -> dict:
        """Log evaluation metrics"""
        
        try:
            # Calculate scores for all metrics
            scores = {
                "code_extraction": 1.0 if report_dict.get("code_extracted") else 0.0,
                "syntax_validity": 1.0 if report_dict.get("syntax_valid") else 0.0,
                "test_pass_rate": (report_dict.get("tests_run", 0) - report_dict.get("tests_failed", 0)) / max(1, report_dict.get("tests_run", 1)),
                "overall_success": {
                    "PASSED": 1.0,
                    "COMPLETED": 0.5,
                    "ERROR": 0.0
                }.get(report_dict.get("overall_status", "ERROR"), 0.0),
                "time_efficiency": min(1.0, max(0.0, 1.0 - (report_dict.get("duration_seconds", TIME_BASELINE_S + TIME_SCALE_S) - TIME_BASELINE_S) / TIME_SCALE_S)),
            }
            
            # Log entry
            log_entry = {
                "model_name": model_name,
                "mode": mode,
                "run_num": run_num,
                "task_id": report_dict.get("task_id"),
                "scores": scores,
                "timestamp": datetime.now().isoformat(),
            }
            self.metrics_log.append(log_entry)
            
            return scores
        except Exception as e:
            print(f"  Warning: Evaluation logging failed: {e}")
            return {}
    
    def get_summary_stats(self, all_reports: list) -> dict:
        """Generate summary statistics for thesis"""
        
        summary = {}
        grouped = defaultdict(list)
        
        # Group by model and mode
        for report in all_reports:
            key = (report.get("model_name"), report.get("mode"))
            grouped[key].append(report)
        
        # Calculate stats per group
        for (model, mode), reports in grouped.items():
            group_key = f"{model}_{mode}"
            
            passed = sum(1 for r in reports if r.get("overall_status") == "PASSED")
            completed = sum(1 for r in reports if r.get("overall_status") == "COMPLETED")
            failed = sum(1 for r in reports if r.get("overall_status") == "ERROR")
            
            avg_time = sum(r.get("duration_seconds", 0) for r in reports) / len(reports) if reports else 0
            avg_tests_passed = sum(r.get("tests_run", 0) - r.get("tests_failed", 0) for r in reports) / len(reports) if reports else 0
            
            success_rate = (passed + completed) / len(reports) if reports else 0
            
            summary[group_key] = {
                "passed": passed,
                "completed": completed,
                "failed": failed,
                "total": len(reports),
                "success_rate": success_rate,
                "avg_time": avg_time,
                "avg_tests_passed": avg_tests_passed,
            }
        
        return summary
    
    def print_summary(self):
        """Print evaluation summary grouped by model and mode."""
        print("\n" + "=" * 80)
        print("  EVALUATION SUMMARY - MAS vs SINGLE")
        print("=" * 80)

        if not self.metrics_log:
            print("  No evaluation data logged yet.")
            return

        grouped = defaultdict(list)
        for entry in self.metrics_log:
            grouped[(entry["model_name"], entry["mode"])].append(entry)

        for (model, mode), entries in sorted(grouped.items()):
            scores = [e["scores"] for e in entries]
            avg = lambda key: sum(s.get(key, 0) for s in scores) / len(scores) if scores else 0

            print(f"\n  {model.upper()} | {mode.upper()}")
            print(f"  {'─'*40}")
            print(f"    Tasks logged      : {len(entries)}")
            print(f"    Code extraction   : {avg('code_extraction'):.1%}")
            print(f"    Syntax validity   : {avg('syntax_validity'):.1%}")
            print(f"    Test pass rate    : {avg('test_pass_rate'):.1%}")
            print(f"    Overall success   : {avg('overall_success'):.1%}")
            print(f"    Time efficiency   : {avg('time_efficiency'):.1%}")

        print("=" * 80)
