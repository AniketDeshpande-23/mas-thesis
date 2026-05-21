"""
analytics/_loader.py — shared CSV loader for all analytics scripts.

Loads and merges all results CSVs, normalises columns, and adds run labels.
"""
from __future__ import annotations
import glob, os, csv
from pathlib import Path
import pandas as pd

# Canonical run labels (timestamp → human label)
RUN_LABELS: dict[str, str] = {
    "20260429_093220": "Baseline (no reforms)",
    "20260429_220044": "Single-Crew refactor",
    "20260430_084337": "Run A — qwen3.5-9b baseline",
    "20260430_184639": "Run A (resume)",
    "20260502_093923": "Run C — convergence loop (partial)",
    "20260509_111330": "Run E — Gemini tools baseline",
    "20260509_134419": "Run E2 — qwen3-coder-next no tools",
    "20260509_174329": "Run F — Reforms 2+3+5",
    "20260509_221652": "Run G — Reform 6 Planner tools",
    "20260513_133605": "Run H — Reform 7 Devstral hard",
    "20260513_140944": "Run H2",
    "20260513_163022": "Run I — Reform 8 max_iter",
    "20260513_173507": "Run J — Claude Opus Planner",
    "20260513_190709": "Run K — Qwen3CoderNext Developer ★ MAS WIN",
    "20260513_202538": "Run M — Fix1+Fix2 ★ MAS WIN",
}

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_all(min_patch_score_col: bool = True) -> pd.DataFrame:
    """Load and merge all results CSVs. Returns combined DataFrame."""
    frames = []
    for path in sorted(RESULTS_DIR.glob("results_*.csv")):
        ts = path.stem.replace("results_", "")
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        df["run_ts"]    = ts
        df["run_label"] = RUN_LABELS.get(ts, ts)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No results CSV found in {RESULTS_DIR}")

    combined = pd.concat(frames, ignore_index=True)

    # Numeric coercion
    for col in ["patch_score", "file_recall", "content_overlap", "codebleu",
                "debug_improvement", "initial_patch_score", "duration_seconds",
                "llm_calls", "total_latency_ms", "avg_latency_ms",
                "debug_iterations", "gold_patch_size", "gen_files_count"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Boolean coercion
    for col in ["docker_resolved", "patch_changed_by_debug",
                "tester_approved", "reviewer_approved", "syntax_valid", "code_extracted"]:
        if col in combined.columns:
            combined[col] = combined[col].map(
                lambda x: str(x).strip().lower() in ("true", "1", "yes") if pd.notna(x) else False
            )

    return combined


def latest_run(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the most recent run (by timestamp)."""
    latest_ts = df["run_ts"].max()
    return df[df["run_ts"] == latest_ts].copy()


def key_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the labelled key experimental runs."""
    return df[df["run_ts"].isin(RUN_LABELS)].copy()
