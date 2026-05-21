"""
analytics/stats.py — Statistical significance evaluation for MAS vs Single LLM thesis.

Computes: t-test, Mann-Whitney U, Cohen's d, bootstrap 95% CIs, per-difficulty breakdown.
Usage: python analytics/stats.py [--csv path/to/results.csv]
"""
from __future__ import annotations
import argparse, warnings
import numpy as np
import pandas as pd
from scipy import stats as sp

warnings.filterwarnings("ignore")

from analytics._loader import load_all, latest_run, key_runs


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    boots = [np.random.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def effect_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:  return "negligible"
    if d < 0.5:  return "small"
    if d < 0.8:  return "medium"
    return "large"


def run_tests(df: pd.DataFrame, metric: str = "patch_score", label: str = "") -> None:
    df = df.dropna(subset=[metric, "mode"])
    mas    = df[df["mode"] == "mas"][metric].values
    single = df[df["mode"] == "single"][metric].values

    if len(mas) < 2 or len(single) < 2:
        print(f"  [SKIP] {label}: insufficient data (MAS n={len(mas)}, Single n={len(single)})")
        return

    t_stat, t_p   = sp.ttest_ind(mas, single, equal_var=False)
    u_stat, u_p   = sp.mannwhitneyu(mas, single, alternative="two-sided")
    d             = cohens_d(mas, single)
    ci_mas        = bootstrap_ci(mas)
    ci_single     = bootstrap_ci(single)

    print(f"\n{'-'*60}")
    print(f"  {label}  (metric: {metric})")
    print(f"{'-'*60}")
    print(f"  MAS    n={len(mas):3d}  mean={mas.mean():.4f}  std={mas.std():.4f}  "
          f"95% CI [{ci_mas[0]:.4f}, {ci_mas[1]:.4f}]")
    print(f"  Single n={len(single):3d}  mean={single.mean():.4f}  std={single.std():.4f}  "
          f"95% CI [{ci_single[0]:.4f}, {ci_single[1]:.4f}]")
    print(f"  Δ (MAS - Single) = {mas.mean() - single.mean():+.4f}")
    print(f"  t-test:  t={t_stat:+.3f}, p={t_p:.4f}  {'*** p<0.01' if t_p<0.01 else '** p<0.05' if t_p<0.05 else '* p<0.10' if t_p<0.10 else '(n.s.)'}")
    print(f"  Mann-Whitney U:  U={u_stat:.0f}, p={u_p:.4f}  {'*** p<0.01' if u_p<0.01 else '** p<0.05' if u_p<0.05 else '* p<0.10' if u_p<0.10 else '(n.s.)'}")
    print(f"  Cohen's d = {d:+.3f} ({effect_label(d)} effect, {'MAS better' if d>0 else 'Single better'})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Specific CSV file to analyse")
    parser.add_argument("--all-runs", action="store_true", help="Include all historical runs")
    args = parser.parse_args()

    np.random.seed(42)
    df_all = load_all()

    if args.csv:
        df = pd.read_csv(args.csv, low_memory=False)
        for col in ["patch_score","file_recall","debug_improvement","duration_seconds","llm_calls"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    elif args.all_runs:
        df = key_runs(df_all)
    else:
        df = latest_run(df_all)

    print(f"\n{'='*60}")
    print("  STATISTICAL SIGNIFICANCE REPORT")
    print(f"  Rows analysed: {len(df)} | Runs: {df['run_ts'].nunique() if 'run_ts' in df.columns else 1}")
    print(f"{'='*60}")

    # ── Overall ───────────────────────────────────────────────────
    run_tests(df, "patch_score",      "Overall — patch_score")
    run_tests(df, "file_recall",      "Overall — file_recall")
    run_tests(df, "debug_improvement","Overall — debug_improvement")

    # ── Per difficulty ────────────────────────────────────────────
    if "difficulty" in df.columns:
        for diff in ["easy", "medium", "hard"]:
            sub = df[df["difficulty"] == diff]
            if len(sub) >= 4:
                run_tests(sub, "patch_score", f"Difficulty={diff} — patch_score")

    # ── Per language ──────────────────────────────────────────────
    if "language" in df.columns:
        for lang in df["language"].dropna().unique():
            sub = df[df["language"] == lang]
            if len(sub) >= 4:
                run_tests(sub, "patch_score", f"Language={lang} — patch_score")

    # ── Compute efficiency ────────────────────────────────────────
    if "llm_calls" in df.columns:
        df2 = df.copy()
        df2["efficiency"] = df2["patch_score"] / df2["llm_calls"].replace(0, np.nan)
        run_tests(df2.dropna(subset=["efficiency"]), "efficiency",
                  "Cost-efficiency (patch_score / llm_calls)")

    print(f"\n{'='*60}")
    print()


if __name__ == "__main__":
    main()
