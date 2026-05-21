"""
analytics/emergent.py — Emergent behaviour and convergence curve analysis.

Analyses whether the debug loop converges toward gold patches and whether
MAS exhibits collaborative error recovery across iterations.

Produces:
  - debug_improvement distribution (MAS vs Single)
  - patch_changed_by_debug rate by mode × difficulty
  - debug_iterations distribution
  - tester_pass_iteration histogram

Usage: python analytics/emergent.py
Output: analytics/out/emergent_*.html
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics._loader import load_all, latest_run

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)


def debug_improvement_chart(df: pd.DataFrame) -> go.Figure:
    df2 = df.dropna(subset=["debug_improvement", "mode"]).copy()
    df2["debug_improvement"] = pd.to_numeric(df2["debug_improvement"], errors="coerce")
    df2 = df2.dropna(subset=["debug_improvement"])

    fig = px.box(
        df2, x="mode", y="debug_improvement", color="mode",
        facet_col="difficulty" if "difficulty" in df2.columns else None,
        color_discrete_map={"mas": "#2E74B5", "single": "#E74C3C"},
        points="all",
        title="Debug Loop Improvement (patch_score delta: final − initial)",
        labels={"debug_improvement": "debug_improvement (Δ patch_score)", "mode": ""},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="No change")
    fig.update_layout(plot_bgcolor="white", font=dict(family="Arial", size=11), height=420)
    return fig


def patch_changed_chart(df: pd.DataFrame) -> go.Figure:
    if "patch_changed_by_debug" not in df.columns:
        return go.Figure()

    df2 = df.copy()
    df2["changed"] = df2["patch_changed_by_debug"].map(
        lambda x: str(x).lower() in ("true", "1", "yes")
    )
    grp = (df2.groupby(["mode", "difficulty"])["changed"]
              .agg(["sum", "count"])
              .reset_index())
    grp["rate"] = 100 * grp["sum"] / grp["count"]
    grp["label"] = grp["mode"].str.upper() + " / " + grp["difficulty"]

    fig = px.bar(
        grp, x="label", y="rate", color="mode",
        color_discrete_map={"mas": "#2E74B5", "single": "#E74C3C"},
        text=grp["rate"].round(0).astype(int).astype(str) + "%",
        title="Patch Changed by Debug Loop (% of runs where debug altered the patch)",
        labels={"rate": "% runs where patch changed", "label": "Mode / Difficulty"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(plot_bgcolor="white", font=dict(family="Arial", size=11),
                      showlegend=True, height=400, yaxis=dict(range=[0, 105]))
    return fig


def debug_iterations_chart(df: pd.DataFrame) -> go.Figure:
    if "debug_iterations" not in df.columns:
        return go.Figure()
    df2 = df.dropna(subset=["debug_iterations", "mode"]).copy()
    df2["debug_iterations"] = pd.to_numeric(df2["debug_iterations"], errors="coerce")

    fig = px.histogram(
        df2, x="debug_iterations", color="mode", barmode="overlay",
        color_discrete_map={"mas": "#2E74B5", "single": "#E74C3C"},
        opacity=0.7,
        title="Distribution of Debug Iterations (0 = no debug loop ran)",
        labels={"debug_iterations": "Number of debug iterations", "mode": ""},
        nbins=8,
    )
    fig.update_layout(plot_bgcolor="white", font=dict(family="Arial", size=11), height=380)
    return fig


def convergence_by_difficulty(df: pd.DataFrame) -> go.Figure:
    """
    Proxy convergence curve: initial_patch_score vs final patch_score,
    grouped by mode and difficulty. Shows whether the debug loop moves
    the patch toward or away from the gold patch.
    """
    needed = {"initial_patch_score", "patch_score", "mode"}
    if not needed.issubset(df.columns):
        return go.Figure()

    df2 = df.dropna(subset=list(needed)).copy()
    for c in ["initial_patch_score", "patch_score"]:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.dropna(subset=list(needed))

    rows, cols = 1, 3
    diffs = ["easy", "medium", "hard"] if "difficulty" in df2.columns else ["all"]
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[d.capitalize() for d in diffs],
                        shared_yaxes=True)

    colors = {"mas": "#2E74B5", "single": "#E74C3C"}
    for ci, diff in enumerate(diffs, 1):
        sub = df2[df2["difficulty"] == diff] if diff != "all" else df2
        for mode, grp in sub.groupby("mode"):
            x = grp["initial_patch_score"].values
            y = grp["patch_score"].values
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(color=colors.get(mode, "grey"), size=8, opacity=0.7),
                name=mode.upper(), showlegend=(ci == 1),
            ), row=1, col=ci)
        # Diagonal: no improvement line
        rng = [0, 0.85]
        fig.add_trace(go.Scatter(
            x=rng, y=rng, mode="lines",
            line=dict(color="grey", dash="dot", width=1),
            showlegend=(ci == 1), name="No change",
        ), row=1, col=ci)

    fig.update_layout(
        title="Initial vs Final patch_score (points above diagonal = debug loop helped)",
        height=380,
        plot_bgcolor="white",
        font=dict(family="Arial", size=11),
    )
    fig.update_xaxes(title_text="Initial patch_score", range=[0, 0.9])
    fig.update_yaxes(title_text="Final patch_score",   range=[0, 0.9])
    return fig


def main():
    df = load_all()
    df = latest_run(df)

    charts = {
        "emergent_debug_improvement":    debug_improvement_chart(df),
        "emergent_patch_changed":        patch_changed_chart(df),
        "emergent_debug_iterations":     debug_iterations_chart(df),
        "emergent_convergence_scatter":  convergence_by_difficulty(df),
    }

    for name, fig in charts.items():
        path = os.path.join(OUT_DIR, f"{name}.html")
        fig.write_html(path)
        print(f"Saved: {path}")

    # Print summary
    print("\n── Emergent Behaviour Summary ────────────────────────")
    if "debug_improvement" in df.columns:
        for mode, grp in df.groupby("mode"):
            d = pd.to_numeric(grp["debug_improvement"], errors="coerce").dropna()
            pos = (d > 0).sum()
            neg = (d < 0).sum()
            print(f"  {mode.upper():<8} debug_improvement: mean={d.mean():+.4f}  "
                  f"pos={pos}/{len(d)}  neg={neg}/{len(d)}")


if __name__ == "__main__":
    main()
