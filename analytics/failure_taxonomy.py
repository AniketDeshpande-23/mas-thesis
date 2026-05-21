"""
analytics/failure_taxonomy.py — Failure taxonomy analysis and visualisation.

Produces:
  - Stacked bar: failure_mode by mode × difficulty
  - Sankey: task → failure_mode → pattern_matched
  - Summary table printed to console

Usage: python analytics/failure_taxonomy.py
Output: analytics/out/failure_taxonomy.html
"""
from __future__ import annotations
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from analytics._loader import load_all, latest_run, key_runs

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)

FAILURE_ORDER = ["ok", "wrong_file", "zero_content", "no_extraction", "error"]
FAILURE_COLORS = {
    "ok":           "#2E8B57",
    "wrong_file":   "#E74C3C",
    "zero_content": "#E67E22",
    "no_extraction":"#9B59B6",
    "error":        "#95A5A6",
}


def stacked_bar(df: pd.DataFrame) -> go.Figure:
    if "failure_mode" not in df.columns or "difficulty" not in df.columns:
        return go.Figure()

    df2 = df.copy()
    df2["failure_mode"] = df2["failure_mode"].fillna("error")
    df2["difficulty"]   = pd.Categorical(df2["difficulty"], ["easy", "medium", "hard"])
    df2["group"]        = df2["mode"].str.upper() + " / " + df2["difficulty"].astype(str)

    counts = (df2.groupby(["group", "failure_mode"])
                 .size()
                 .reset_index(name="count"))
    totals = counts.groupby("group")["count"].transform("sum")
    counts["pct"] = 100 * counts["count"] / totals

    fig = px.bar(
        counts, x="group", y="pct", color="failure_mode",
        color_discrete_map=FAILURE_COLORS,
        category_orders={"failure_mode": FAILURE_ORDER},
        labels={"pct": "% of runs", "group": "Mode / Difficulty", "failure_mode": "Failure Mode"},
        title="Failure Mode Distribution by Architecture and Task Difficulty",
        text="count",
    )
    fig.update_layout(
        barmode="stack",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12),
        legend_title="Failure Mode",
        height=480,
    )
    fig.update_traces(textposition="inside", textfont_size=10)
    return fig


def sankey(df: pd.DataFrame) -> go.Figure:
    if "failure_mode" not in df.columns:
        return go.Figure()

    df2 = df.copy()
    df2["failure_mode"]   = df2["failure_mode"].fillna("error")
    df2["pattern_simple"] = df2["pattern_matched"].fillna("none").apply(
        lambda x: x.split(",")[0].strip() if x and x != "none" else "none"
    )
    df2["mode_label"] = df2["mode"].str.upper()

    modes    = sorted(df2["mode_label"].unique())
    failures = sorted(df2["failure_mode"].unique())
    patterns = sorted(df2["pattern_simple"].unique())

    labels = modes + failures + patterns
    label_idx = {l: i for i, l in enumerate(labels)}

    sources, targets, values = [], [], []

    # mode → failure_mode
    for (m, f), grp in df2.groupby(["mode_label", "failure_mode"]):
        sources.append(label_idx[m])
        targets.append(label_idx[f])
        values.append(len(grp))

    # failure_mode → pattern
    for (f, p), grp in df2.groupby(["failure_mode", "pattern_simple"]):
        sources.append(label_idx[f])
        targets.append(label_idx[p])
        values.append(len(grp))

    node_colors = []
    for l in labels:
        if l in ("MAS", "SINGLE"):
            node_colors.append("#2E74B5")
        elif l in FAILURE_COLORS:
            node_colors.append(FAILURE_COLORS[l])
        else:
            node_colors.append("#BDC3C7")

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, color=node_colors, pad=15, thickness=20),
        link=dict(source=sources, target=targets, value=values,
                  color="rgba(180,180,180,0.4)"),
    ))
    fig.update_layout(
        title="Failure Flow: Mode → Failure Type → Bug Pattern",
        font=dict(family="Arial", size=11),
        height=500,
    )
    return fig


def print_summary(df: pd.DataFrame) -> None:
    if "failure_mode" not in df.columns:
        print("failure_mode column not found")
        return
    print("\n── Failure Mode Summary ──────────────────────────────")
    tbl = (df.groupby(["mode", "failure_mode"])
             .size()
             .reset_index(name="n")
             .pivot(index="failure_mode", columns="mode", values="n")
             .fillna(0)
             .astype(int))
    print(tbl.to_string())

    print("\n── Pattern Matched (top 5) ───────────────────────────")
    if "pattern_matched" in df.columns:
        pat = (df["pattern_matched"].fillna("none")
                 .str.split(",")
                 .explode()
                 .str.strip()
                 .value_counts()
                 .head(10))
        print(pat.to_string())


def main():
    df_all  = load_all()
    df      = latest_run(df_all)

    print(f"Rows: {len(df)}  |  Runs: {df['run_ts'].nunique()}")
    print_summary(df)

    bar  = stacked_bar(df)
    sank = sankey(df)

    combined = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Failure Mode Distribution", "Failure Flow Sankey"),
        specs=[[{"type": "bar"}, {"type": "sankey"}]],
        column_widths=[0.55, 0.45],
    )

    for trace in bar.data:
        combined.add_trace(trace, row=1, col=1)
    for trace in sank.data:
        combined.add_trace(trace, row=1, col=2)

    combined.update_layout(
        title="Failure Taxonomy Analysis — MAS vs Single LLM",
        height=520,
        barmode="stack",
        plot_bgcolor="white",
        font=dict(family="Arial", size=11),
        showlegend=True,
    )

    out = os.path.join(OUT_DIR, "failure_taxonomy.html")
    combined.write_html(out)
    print(f"\nSaved: {out}")

    # Also save individual charts for thesis figures
    bar.write_html(os.path.join(OUT_DIR, "failure_taxonomy_bar.html"))
    sank.write_html(os.path.join(OUT_DIR, "failure_sankey.html"))


if __name__ == "__main__":
    main()
