"""
analytics/patch_quality.py — Patch quality analytics.

Analyses patch size, edit locality, file targeting, and similarity to gold patch.

Usage: python analytics/patch_quality.py
Output: analytics/out/patch_quality_*.html
"""
from __future__ import annotations
import os, json, glob, re
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics._loader import load_all, latest_run

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
RESULTS_DIR = Path(__file__).parent.parent / "results"
os.makedirs(OUT_DIR, exist_ok=True)


def _patch_stats(patch: str) -> dict:
    """Compute hunk count, additions, deletions from a raw unified diff."""
    if not patch or not isinstance(patch, str):
        return {"hunks": 0, "additions": 0, "deletions": 0, "files": 0}
    lines = patch.split("\n")
    return {
        "hunks":     sum(1 for l in lines if l.startswith("@@")),
        "additions": sum(1 for l in lines if l.startswith("+") and not l.startswith("+++")),
        "deletions": sum(1 for l in lines if l.startswith("-") and not l.startswith("---")),
        "files":     sum(1 for l in lines if l.startswith("diff --git")),
    }


def enrich_with_patch_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Load patch JSON files and attach gen_hunks, gen_additions, gen_deletions."""
    patch_files = sorted(RESULTS_DIR.glob("swebench_patches_*.csv"))
    # Build instance_id → patch map from JSON files
    patch_map: dict[str, str] = {}
    for pf in sorted(RESULTS_DIR.glob("swebench_patches_*.json")):
        try:
            with open(pf) as f:
                for rec in json.load(f):
                    iid = rec.get("instance_id", "")
                    if iid and iid not in patch_map:
                        patch_map[iid] = rec.get("patch", "")
        except Exception:
            pass

    if not patch_map:
        return df

    stats = df["task_id"].map(lambda tid: _patch_stats(patch_map.get(tid, "")))
    df = df.copy()
    df["gen_hunks"]     = stats.map(lambda s: s["hunks"])
    df["gen_additions"] = stats.map(lambda s: s["additions"])
    df["gen_deletions"] = stats.map(lambda s: s["deletions"])
    return df


def edit_locality_chart(df: pd.DataFrame) -> go.Figure:
    """gen_hunks / gold_hunks ratio — 1.0 = perfectly surgical, >1 = over-patching."""
    needed = {"gen_hunks", "gold_hunks", "mode"}
    if not needed.issubset(df.columns):
        return go.Figure()

    df2 = df.copy()
    for c in ["gen_hunks", "gold_hunks"]:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.dropna(subset=["gen_hunks", "gold_hunks", "mode"])
    df2 = df2[df2["gold_hunks"] > 0]
    df2["locality"] = df2["gen_hunks"] / df2["gold_hunks"]

    fig = px.box(
        df2, x="mode", y="locality", color="mode",
        color_discrete_map={"mas": "#2E74B5", "single": "#E74C3C"},
        points="all",
        title="Edit Locality (gen_hunks / gold_hunks — 1.0 = perfectly targeted)",
        labels={"locality": "Hunk ratio (gen / gold)", "mode": ""},
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Perfect match")
    fig.update_layout(plot_bgcolor="white", font=dict(family="Arial", size=11), height=400)
    return fig


def size_vs_score_chart(df: pd.DataFrame) -> go.Figure:
    needed = {"gold_patch_size", "patch_score", "mode", "failure_mode"}
    avail  = needed & set(df.columns)
    if "gold_patch_size" not in avail or "patch_score" not in avail:
        return go.Figure()

    df2 = df.copy()
    df2["gold_patch_size"] = pd.to_numeric(df2["gold_patch_size"], errors="coerce")
    df2["patch_score"]     = pd.to_numeric(df2["patch_score"],     errors="coerce")
    df2 = df2.dropna(subset=["gold_patch_size", "patch_score"])

    color_col = "failure_mode" if "failure_mode" in df2.columns else "mode"
    fig = px.scatter(
        df2, x="gold_patch_size", y="patch_score",
        color=color_col, symbol="mode",
        trendline="ols",
        title="Gold Patch Size vs Achieved patch_score (by failure mode)",
        labels={"gold_patch_size": "Gold patch size (chars)", "patch_score": "patch_score"},
        hover_data=["task_id", "mode", "difficulty"] if "difficulty" in df2.columns else ["task_id", "mode"],
    )
    fig.update_layout(plot_bgcolor="white", font=dict(family="Arial", size=11), height=440)
    return fig


def quality_heatmap(df: pd.DataFrame) -> go.Figure:
    metrics = ["patch_score", "file_recall", "content_overlap", "codebleu"]
    avail   = [m for m in metrics if m in df.columns]
    if not avail or "mode" not in df.columns:
        return go.Figure()

    df2 = df.copy()
    if "difficulty" in df2.columns:
        grp = df2.groupby(["mode", "difficulty"])[avail].mean().reset_index()
        grp["label"] = grp["mode"].str.upper() + " / " + grp["difficulty"]
    else:
        grp = df2.groupby("mode")[avail].mean().reset_index()
        grp["label"] = grp["mode"].str.upper()

    z_vals = grp[avail].values
    fig = go.Figure(go.Heatmap(
        z=z_vals.T,
        x=grp["label"].tolist(),
        y=avail,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in z_vals.T],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title="Patch Quality Metrics Heatmap (Mode × Difficulty)",
        font=dict(family="Arial", size=11),
        height=300,
        xaxis=dict(tickangle=-20),
    )
    return fig


def main():
    df = load_all()
    df = latest_run(df)
    df = enrich_with_patch_stats(df)

    charts = {
        "patch_quality_locality":  edit_locality_chart(df),
        "patch_quality_size_score":size_vs_score_chart(df),
        "patch_quality_heatmap":   quality_heatmap(df),
    }

    for name, fig in charts.items():
        path = os.path.join(OUT_DIR, f"{name}.html")
        fig.write_html(path)
        print(f"Saved: {path}")

    # Console summary
    print("\n── Patch Quality Summary ─────────────────────────────")
    metrics = [c for c in ["patch_score","file_recall","content_overlap","codebleu"] if c in df.columns]
    if metrics and "mode" in df.columns:
        tbl = df.groupby("mode")[metrics].mean().round(4)
        print(tbl.to_string())


if __name__ == "__main__":
    main()
