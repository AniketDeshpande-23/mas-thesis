"""
evaluation/pass_at_k.py

Pass@k and Resolved Rate metrics for the MAS vs Single thesis experiment.

Pass@k  (Chen et al. 2021, unbiased estimator)
-------
Given n total samples per problem and c correct samples:
    pass@k = 1 - C(n-c, k) / C(n, k)

Averaged across all problems in the benchmark.

With NUM_RUNS=1  → only pass@1 is valid (equals resolved_rate).
With NUM_RUNS=3  → pass@1, pass@2, pass@3 can all be estimated.

Resolved Rate
-------------
Fraction of (task, run) pairs where overall_status in {PASSED, COMPLETED}.
Reported per (model, mode) condition.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple


_RESOLVED = {"PASSED", "COMPLETED"}


def _pass_at_k_single(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem: n samples, c correct, pick k."""
    if n < k:
        return float("nan")          # not enough samples to estimate
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_metrics(
    reports: List[dict],
    k_values: List[int] | None = None,
) -> Dict[Tuple[str, str], dict]:
    """
    Compute resolved_rate and pass@k for every (model_name, mode) condition.

    Parameters
    ----------
    reports   : list of report dicts from main.py (one per task × run)
    k_values  : list of k values to compute; defaults to [1, 2, 3]

    Returns
    -------
    dict keyed by (model_name, mode) →
        {
          "resolved_rate": float,
          "pass@1": float,
          "pass@2": float,   # nan if fewer than 2 runs per task
          "pass@3": float,   # nan if fewer than 3 runs per task
          "n_tasks": int,
          "n_runs":  int,    # max runs seen per task
        }
    """
    if k_values is None:
        k_values = [1, 2, 3]

    # Group outcomes: {(model, mode) → {task_id → [bool, ...]}}
    groups: Dict[Tuple[str, str], Dict[str, List[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in reports:
        key = (r.get("model_name", ""), r.get("mode", ""))
        task = r.get("task_id", "")
        resolved = r.get("overall_status", "") in _RESOLVED
        groups[key][task].append(resolved)

    results: Dict[Tuple[str, str], dict] = {}
    for (model, mode), task_outcomes in groups.items():
        total_pairs   = sum(len(v) for v in task_outcomes.values())
        total_resolved = sum(sum(v) for v in task_outcomes.values())
        resolved_rate = round(total_resolved / total_pairs, 4) if total_pairs else 0.0

        max_runs = max((len(v) for v in task_outcomes.values()), default=0)

        pass_scores: Dict[str, float] = {}
        for k in k_values:
            valid = [
                _pass_at_k_single(len(v), sum(v), k)
                for v in task_outcomes.values()
                if not math.isnan(_pass_at_k_single(len(v), sum(v), k))
            ]
            pass_scores[f"pass@{k}"] = round(sum(valid) / len(valid), 4) if valid else float("nan")

        results[(model, mode)] = {
            "resolved_rate": resolved_rate,
            **pass_scores,
            "n_tasks": len(task_outcomes),
            "n_runs":  max_runs,
        }

    return results


def format_summary(metrics: Dict[Tuple[str, str], dict]) -> str:
    """Return a formatted multi-line string for printing in the run summary."""
    lines = []
    for (model, mode), m in sorted(metrics.items()):
        lines.append(f"  {model.upper()} | {mode.upper()}")
        lines.append(f"    Resolved rate : {m['resolved_rate']:.1%}  ({m['n_tasks']} tasks × {m['n_runs']} run(s))")
        for k in [1, 2, 3]:
            val = m.get(f"pass@{k}", float("nan"))
            if not math.isnan(val):
                lines.append(f"    Pass@{k}        : {val:.4f}")
        lines.append("")
    return "\n".join(lines)
