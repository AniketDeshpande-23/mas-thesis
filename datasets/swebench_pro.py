"""
SWE-bench Pro dataset handler  (thin wrapper)
==============================================

This file mirrors datasets/bigcodebench_hard.py which re-exports
BigCodeBenchHard from data_loaders/bigcodebench_hard.py.

"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Re-export so imports from either path work
from data_loaders.swebench_pro import (        # noqa: F401
    SWEBenchPro,
    SWEBenchProInstance,
    Difficulty,
    TEST_SUBSET_IDS,
)

logger = logging.getLogger(__name__)


# ── Evaluation result for one instance ───────────────────────────────

@dataclass
class EvalResult:
    """Result for a single evaluated instance."""
    instance_id: str
    resolved: bool
    fail_to_pass_passed: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_passed: int = 0
    pass_to_pass_total: int = 0
    generated_patch: str = ""
    error_message: str = ""
    time_seconds: float = 0.0
    tokens_used: int = 0
    agent_iterations: int = 0
    model_name: str = ""


# ── Patch formatting helpers ─────────────────────────────────────────

def format_patches_for_eval(
    patches: Dict[str, str], prefix: str = "model"
) -> List[Dict]:
    """
    Convert {instance_id: patch_str} to the JSON format expected
    by the official swe_bench_pro_eval.py script.
    """
    return [
        {"instance_id": iid, "patch": patch, "prefix": prefix}
        for iid, patch in patches.items()
    ]


def save_patches(
    patches: Dict[str, str],
    output_path: str,
    prefix: str = "model",
):
    """Save patches in the format expected by the official eval script."""
    formatted = format_patches_for_eval(patches, prefix)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(formatted, f, indent=2)
    logger.info("Saved %d patches to %s", len(formatted), output_path)


# ── Results aggregation ──────────────────────────────────────────────

def aggregate_results(results: List[EvalResult]) -> Dict:
    """
    Aggregate evaluation results into summary metrics matching the
    thesis evaluation criteria:
      - resolve_rate (overall and per-repo)
      - execution_time stats
      - token_usage stats
      - agent_iteration stats
    """
    if not results:
        return {}

    resolved = [r for r in results if r.resolved]
    total = len(results)

    # Per-repo
    by_repo: Dict[str, List[EvalResult]] = {}
    for r in results:
        iid = r.instance_id
        repo = (
            iid.split("__")[0].replace("instance_", "").replace("_", "/", 1)
            if "__" in iid
            else "unknown"
        )
        by_repo.setdefault(repo, []).append(r)

    repo_rates = {
        repo: sum(1 for r in rs if r.resolved) / len(rs)
        for repo, rs in by_repo.items()
    }

    times = [r.time_seconds for r in results if r.time_seconds > 0]
    tokens = [r.tokens_used for r in results if r.tokens_used > 0]
    iterations = [r.agent_iterations for r in results if r.agent_iterations > 0]

    def _stats(values):
        if not values:
            return {"mean": 0, "median": 0, "min": 0, "max": 0}
        s = sorted(values)
        return {
            "mean": sum(s) / len(s),
            "median": s[len(s) // 2],
            "min": s[0],
            "max": s[-1],
        }

    return {
        "resolve_rate": len(resolved) / total,
        "resolved_count": len(resolved),
        "total_count": total,
        "by_repo": repo_rates,
        "time_stats": _stats(times),
        "token_stats": _stats(tokens),
        "iteration_stats": _stats(iterations),
        "error_count": sum(1 for r in results if r.error_message),
    }
