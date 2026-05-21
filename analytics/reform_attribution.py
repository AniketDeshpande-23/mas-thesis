"""
analytics/reform_attribution.py — Reform attribution waterfall chart.

Shows how each architectural reform contributed to closing the MAS-Single gap.
Data is hardcoded from documented run results (E through M).

Usage: python analytics/reform_attribution.py
Output: analytics/out/reform_attribution.html
"""
from __future__ import annotations
import os
import plotly.graph_objects as go

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)

# (run_label, reform_added, MAS_score, Single_score)
RUN_DATA = [
    ("Run E\nBaseline (tools)",          "No reforms",                        0.251, 0.353),
    ("Run F\nReforms 2+3+5",             "Delta signal + Jaccard + Adaptive", 0.230, 0.323),
    ("Run G\nReform 6",                  "Planner gets RepoSearch tools",     0.247, 0.293),
    ("Run H\nReform 7",                  "Devstral hard + Planner 3-call cap",0.224, 0.230),
    ("Run I\nReform 8",                  "Developer max_iter adaptive",       0.234, 0.272),
    ("Run J\nOpus Planner",              "Claude Opus as Planner+Reviewer",   0.228, 0.292),
    ("Run K\nQwen3 Developer ★",         "Qwen3-Coder-Next as Developer",     0.334, 0.278),
    ("Run M\nFix 1+2 ★",                "Adaptive budget + zero-score fix",  0.336, 0.273),
]


def waterfall_chart() -> go.Figure:
    labels  = [r[0] for r in RUN_DATA]
    gaps    = [round(r[2] - r[3], 4) for r in RUN_DATA]  # MAS - Single (positive = MAS wins)
    reforms = [r[1] for r in RUN_DATA]

    colors = ["#E74C3C" if g < 0 else "#2E8B57" for g in gaps]

    fig = go.Figure()

    # Gap bars
    fig.add_trace(go.Bar(
        x=labels, y=gaps,
        marker_color=colors,
        text=[f"{g:+.3f}" for g in gaps],
        textposition="outside",
        name="MAS − Single gap",
        hovertemplate="<b>%{x}</b><br>Gap: %{y:+.3f}<br>Reform: %{customdata}<extra></extra>",
        customdata=reforms,
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1.5)

    # Annotation: first MAS win
    fig.add_annotation(
        x="Run K\nQwen3 Developer ★",
        y=0.056,
        text="First MAS Win",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#2E8B57",
        font=dict(color="#2E8B57", size=11, family="Arial Bold"),
        ax=40, ay=-30,
    )

    fig.update_layout(
        title="Reform Attribution: MAS−Single Gap Across Experiment Runs",
        xaxis_title="Experiment Run",
        yaxis_title="MAS − Single (patch_score)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        height=460,
        showlegend=False,
        yaxis=dict(zeroline=False, gridcolor="#ECECEC"),
        xaxis=dict(tickangle=-15),
    )
    return fig


def mas_single_line() -> go.Figure:
    labels  = [r[0] for r in RUN_DATA]
    mas     = [r[2] for r in RUN_DATA]
    single  = [r[3] for r in RUN_DATA]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=mas, mode="lines+markers",
        name="MAS", line=dict(color="#2E74B5", width=2.5),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=single, mode="lines+markers",
        name="Single", line=dict(color="#E74C3C", width=2.5, dash="dash"),
        marker=dict(size=8),
    ))

    # Shade the win zone (K onward)
    fig.add_vrect(
        x0="Run K\nQwen3 Developer ★", x1="Run M\nFix 1+2 ★",
        fillcolor="#2E8B57", opacity=0.08,
        annotation_text="MAS leads", annotation_position="top left",
        annotation_font=dict(color="#2E8B57", size=10),
    )

    fig.update_layout(
        title="MAS vs Single patch_score Progression Across Reforms",
        xaxis_title="Experiment Run",
        yaxis_title="Average patch_score",
        plot_bgcolor="white",
        font=dict(family="Arial", size=11),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(tickangle=-15),
        yaxis=dict(gridcolor="#ECECEC"),
    )
    return fig


def main():
    wf  = waterfall_chart()
    lc  = mas_single_line()

    wf.write_html(os.path.join(OUT_DIR, "reform_attribution_waterfall.html"))
    lc.write_html(os.path.join(OUT_DIR, "reform_attribution_line.html"))

    print("Reform attribution summary:")
    print(f"{'Run':<30} {'MAS':>7} {'Single':>8} {'Gap':>8} {'Winner'}")
    print("─" * 65)
    for label, reform, mas, single in RUN_DATA:
        gap    = mas - single
        winner = "★ MAS" if gap > 0 else "Single"
        print(f"{label.replace(chr(10),' '):<30} {mas:>7.3f} {single:>8.3f} {gap:>+8.3f}  {winner}")

    print(f"\nSaved to analytics/out/")


if __name__ == "__main__":
    main()
