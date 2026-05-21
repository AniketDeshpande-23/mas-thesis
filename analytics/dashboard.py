"""
analytics/dashboard.py — Industry-standard research dashboard.
MAS vs Single LLM Thesis — SWE-bench Pro.

Usage:  streamlit run analytics/dashboard.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp

from analytics._loader import load_all, RUN_LABELS

# ── Palette ───────────────────────────────────────────────────────────────────
C_MAS    = "#1B4F8A"   # deep navy — MAS
C_SINGLE = "#B03A2E"   # deep red  — Single
C_GRID   = "#EAECEE"
C_BG     = "#F7F9FC"
C_WHITE  = "#FFFFFF"
C_TEXT   = "#1C2833"
C_SUB    = "#5D6D7E"
C_WIN    = "#1A7A3C"   # green — MAS win
C_LOSS   = "#B03A2E"   # red   — Single win

FAIL_PALETTE = {
    "ok":           "#1A7A3C",
    "wrong_file":   "#B03A2E",
    "zero_content": "#C87137",
    "no_extraction":"#7D3C98",
    "error":        "#717D7E",
}

REFORM_DATA = [
    ("Run E", "Baseline — tools enabled",                  0.251, 0.353),
    ("Run F", "Delta signal + Jaccard convergence",        0.230, 0.323),
    ("Run G", "Planner given RepoSearch tools",            0.247, 0.293),
    ("Run H", "Devstral for hard tasks, 3-call cap",       0.224, 0.230),
    ("Run I", "Developer max_iter difficulty-adaptive",    0.234, 0.272),
    ("Run J", "Claude Opus as Planner and Reviewer",       0.228, 0.292),
    ("Run K", "Qwen3-Coder-Next as Developer — MAS WIN",   0.334, 0.278),
    ("Run M", "Adaptive budget + zero-score recovery",     0.336, 0.273),
]

MODE_MAP   = {"mas": "MAS", "single": "Single"}
DIFF_ORDER = ["easy", "medium", "hard"]

# ── Global chart layout defaults ──────────────────────────────────────────────
# Base layout — contains ONLY keys that are never passed again as kwargs.
# legend, xaxis, yaxis are intentionally excluded to avoid duplicate-kwarg errors.
_BASE = dict(
    font=dict(family="Inter, Arial, sans-serif", size=12, color=C_TEXT),
    plot_bgcolor=C_WHITE,
    paper_bgcolor=C_WHITE,
    margin=dict(l=40, r=30, t=50, b=40))
_LEGEND_CLEAN  = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0)
_AXIS          = dict(gridcolor=C_GRID, linecolor=C_GRID, zeroline=False)


def _fmt(fig, height=380, legend=None, **extra):
    """Apply base layout + clean legend + axis defaults to any figure."""
    leg = {**_LEGEND_CLEAN, **(legend or {})}
    fig.update_layout(**_BASE, height=height, legend=leg, **extra)
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAS vs Single LLM — Research Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded")

# ── CSS injection ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Global background */
  .stApp {{ background-color: {C_BG}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background-color: #1C2833;
  }}
  [data-testid="stSidebar"] * {{
    color: #D5D8DC !important;
  }}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label {{
    color: #AAB7B8 !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 0px;
    background-color: {C_WHITE};
    border-bottom: 2px solid {C_GRID};
    padding: 0 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    padding: 10px 24px;
    font-size: 0.82rem;
    font-weight: 600;
    color: {C_SUB};
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background-color: transparent;
    border: none;
  }}
  .stTabs [aria-selected="true"] {{
    color: {C_MAS} !important;
    border-bottom: 2px solid {C_MAS};
  }}

  /* KPI cards */
  .kpi-card {{
    background: {C_WHITE};
    border-radius: 6px;
    border-left: 4px solid {C_MAS};
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
  }}
  .kpi-card.win  {{ border-left-color: {C_WIN}; }}
  .kpi-card.loss {{ border-left-color: {C_LOSS}; }}
  .kpi-label {{
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {C_SUB};
    margin-bottom: 4px;
  }}
  .kpi-value {{
    font-size: 1.55rem;
    font-weight: 700;
    color: {C_TEXT};
    line-height: 1;
  }}
  .kpi-delta {{
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 4px;
  }}
  .kpi-delta.pos {{ color: {C_WIN}; }}
  .kpi-delta.neg {{ color: {C_LOSS}; }}

  /* Section headers */
  .section-title {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: {C_SUB};
    border-bottom: 1px solid {C_GRID};
    padding-bottom: 6px;
    margin-bottom: 14px;
    margin-top: 10px;
  }}

  /* Page title */
  .page-header {{
    background: {C_WHITE};
    border-bottom: 1px solid {C_GRID};
    padding: 18px 28px 14px 28px;
    margin: -1rem -1rem 1.5rem -1rem;
  }}
  .page-title {{
    font-size: 1.25rem;
    font-weight: 700;
    color: {C_TEXT};
    letter-spacing: -0.01em;
  }}
  .page-subtitle {{
    font-size: 0.78rem;
    color: {C_SUB};
    margin-top: 2px;
  }}

  /* Hide Streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 0.5rem; }}

  /* Chart containers */
  .chart-card {{
    background: {C_WHITE};
    border-radius: 6px;
    padding: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
  }}

  /* Download button */
  .stDownloadButton > button {{
    background-color: {C_MAS};
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 6px 16px;
    letter-spacing: 0.03em;
  }}
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_data() -> pd.DataFrame:
    return load_all()

df_all = get_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 10px 0'>
      <div style='font-size:0.65rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.12em;color:#717D7E;margin-bottom:2px'>
        RESEARCH DASHBOARD
      </div>
      <div style='font-size:1.0rem;font-weight:700;color:#F0F3F4;line-height:1.2'>
        MAS vs Single LLM<br>Automated Bug Fixing
      </div>
      <div style='font-size:0.70rem;color:#717D7E;margin-top:6px'>
        SWE-bench Pro · patch_score = 0.6 x file_recall + 0.4 x content_overlap
      </div>
    </div>
    <hr style='border-color:#2C3E50;margin:10px 0 16px 0'>
    """, unsafe_allow_html=True)

    run_options  = ["Latest run"] + sorted(df_all["run_ts"].unique(), reverse=True)
    selected_run = st.selectbox("Experiment Run", run_options)

    if selected_run == "Latest run":
        df = df_all[df_all["run_ts"] == df_all["run_ts"].max()].copy()
        run_label = "Run M — Latest"
    else:
        df = df_all[df_all["run_ts"] == selected_run].copy()
        run_label = RUN_LABELS.get(selected_run, selected_run)

    if "difficulty" in df.columns:
        diffs = st.multiselect("Difficulty", DIFF_ORDER, default=DIFF_ORDER)
        df = df[df["difficulty"].isin(diffs)]

    if "language" in df.columns:
        langs = sorted(df["language"].dropna().unique())
        sel_langs = st.multiselect("Language", langs, default=langs)
        df = df[df["language"].isin(sel_langs)]

    st.markdown("<hr style='border-color:#2C3E50;margin:16px 0 10px 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.70rem;color:#AAB7B8'>
      <div><b style='color:#F0F3F4'>{len(df)}</b> pipeline runs</div>
      <div><b style='color:#F0F3F4'>{df["task_id"].nunique() if "task_id" in df.columns else "—"}</b> unique tasks</div>
      <div style='margin-top:6px;color:#717D7E'>{run_label}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
  <div class='page-title'>Automated Bug-Fixing: Multi-Agent System vs Single LLM</div>
  <div class='page-subtitle'>Comparative analysis on SWE-bench Pro &nbsp;|&nbsp; patch_score = 0.6 x file_recall + 0.4 x content_overlap</div>
</div>
""", unsafe_allow_html=True)


# ── KPI helpers ───────────────────────────────────────────────────────────────
def safe_mean(series): return pd.to_numeric(series, errors="coerce").mean()

mas_df    = df[df["mode"] == "mas"]
single_df = df[df["mode"] == "single"]
mas_ps    = safe_mean(mas_df["patch_score"])    if "patch_score"      in df.columns else float("nan")
sng_ps    = safe_mean(single_df["patch_score"]) if "patch_score"      in df.columns else float("nan")
mas_fr    = safe_mean(mas_df["file_recall"])    if "file_recall"      in df.columns else float("nan")
sng_fr    = safe_mean(single_df["file_recall"]) if "file_recall"      in df.columns else float("nan")
mas_di    = safe_mean(mas_df["debug_improvement"]) if "debug_improvement" in df.columns else float("nan")
gap       = mas_ps - sng_ps if not (np.isnan(mas_ps) or np.isnan(sng_ps)) else float("nan")
winner    = "MAS" if gap > 0 else "Single"


def kpi(label, value, delta=None, card_class=""):
    delta_html = ""
    if delta is not None:
        cls = "pos" if delta >= 0 else "neg"
        sign = "+" if delta >= 0 else ""
        delta_html = f"<div class='kpi-delta {cls}'>{sign}{delta:.4f} vs Single</div>"
    return f"""
    <div class='kpi-card {card_class}'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value'>{value}</div>
      {delta_html}
    </div>"""


k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.markdown(kpi("MAS patch_score",    f"{mas_ps:.4f}", delta=gap,
                          card_class="win" if gap > 0 else "loss"), unsafe_allow_html=True)
with k2: st.markdown(kpi("Single patch_score", f"{sng_ps:.4f}"), unsafe_allow_html=True)
with k3:
    gap_class = "win" if gap > 0 else "loss"
    st.markdown(f"""
    <div class='kpi-card {gap_class}'>
      <div class='kpi-label'>Architecture Winner</div>
      <div class='kpi-value' style='font-size:1.2rem'>{winner}</div>
      <div class='kpi-delta {"pos" if gap>0 else "neg"}'>Gap: {gap:+.4f}</div>
    </div>""", unsafe_allow_html=True)
with k4: st.markdown(kpi("MAS file_recall",    f"{mas_fr:.4f}"), unsafe_allow_html=True)
with k5: st.markdown(kpi("MAS debug improvement", f"{mas_di:+.4f}"), unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Performance Comparison",
    "Failure Analysis",
    "Reform Journey",
    "Agent Behaviour",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PERFORMANCE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_sel, _ = st.columns([2, 5])
    with col_sel:
        metric = st.selectbox(
            "Primary metric",
            ["patch_score", "file_recall", "content_overlap", "codebleu", "debug_improvement"],
            key="perf_metric")

    c_left, c_right = st.columns(2)

    # ── Grouped bar by difficulty ─────────────────────────────────────────────
    with c_left:
        st.markdown("<div class='section-title'>Mean Score by Difficulty</div>", unsafe_allow_html=True)
        if "difficulty" in df.columns and metric in df.columns:
            grp = (df.groupby(["difficulty", "mode"])[metric]
                     .mean().reset_index()
                     .rename(columns={"mode": "Architecture"}))
            grp["Architecture"] = grp["Architecture"].map(MODE_MAP)
            grp["difficulty"]   = pd.Categorical(grp["difficulty"], DIFF_ORDER)
            grp = grp.sort_values("difficulty")

            fig = px.bar(
                grp, x="difficulty", y=metric, color="Architecture", barmode="group",
                color_discrete_map={"MAS": C_MAS, "Single": C_SINGLE},
                text=grp[metric].round(3),
                labels={"difficulty": "Task Difficulty", metric: metric})
            fig.update_traces(textposition="outside", textfont_size=11,
                              marker_line_width=0)
            _fmt(fig, height=360)
            fig.update_yaxes(range=[0, grp[metric].max() * 1.18])
            st.plotly_chart(fig, width='stretch')

    # ── Box plot distribution ─────────────────────────────────────────────────
    with c_right:
        st.markdown("<div class='section-title'>Score Distribution</div>", unsafe_allow_html=True)
        if metric in df.columns:
            df2 = df.dropna(subset=[metric]).copy()
            df2["Architecture"] = df2["mode"].map(MODE_MAP)
            fig = px.box(
                df2, x="Architecture", y=metric, color="Architecture",
                color_discrete_map={"MAS": C_MAS, "Single": C_SINGLE},
                points="all",
                labels={"Architecture": "", metric: metric})
            fig.update_traces(
                jitter=0.3, marker=dict(size=5, opacity=0.55),
                line_width=1.5)
            fig.add_hline(y=0, line_dash="dot", line_color=C_GRID, line_width=1)
            _fmt(fig, height=360, showlegend=False)
            st.plotly_chart(fig, width='stretch')

    # ── Per-language bar ──────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Score by Repository Language</div>", unsafe_allow_html=True)
    if "language" in df.columns and metric in df.columns:
        lang_grp = (df.groupby(["language", "mode"])[metric]
                      .mean().reset_index()
                      .rename(columns={"mode": "Architecture"}))
        lang_grp["Architecture"] = lang_grp["Architecture"].map(MODE_MAP)

        fig = px.bar(
            lang_grp, x="language", y=metric, color="Architecture", barmode="group",
            color_discrete_map={"MAS": C_MAS, "Single": C_SINGLE},
            text=lang_grp[metric].round(3),
            labels={"language": "Language", metric: metric})
        fig.update_traces(textposition="outside", textfont_size=11, marker_line_width=0)
        _fmt(fig, height=320)
        fig.update_yaxes(range=[0, lang_grp[metric].max() * 1.2])
        st.plotly_chart(fig, width='stretch')

    # ── Statistical summary card ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>Statistical Summary</div>", unsafe_allow_html=True)
    if metric in df.columns:
        mas_v   = pd.to_numeric(mas_df[metric],    errors="coerce").dropna().values
        sng_v   = pd.to_numeric(single_df[metric], errors="coerce").dropna().values
        if len(mas_v) >= 2 and len(sng_v) >= 2:
            t, p   = sp.ttest_ind(mas_v, sng_v, equal_var=False)
            u, up  = sp.mannwhitneyu(mas_v, sng_v, alternative="two-sided")
            pooled = np.sqrt(((len(mas_v)-1)*mas_v.std(ddof=1)**2 +
                              (len(sng_v)-1)*sng_v.std(ddof=1)**2) /
                             (len(mas_v)+len(sng_v)-2))
            d      = (mas_v.mean() - sng_v.mean()) / pooled if pooled > 0 else 0
            p_str  = f"{p:.4f}" + (" ***" if p<0.01 else " **" if p<0.05 else " *" if p<0.10 else " n.s.")
            up_str = f"{up:.4f}" + (" ***" if up<0.01 else " **" if up<0.05 else " *" if up<0.10 else " n.s.")
            d_lbl  = ("large" if abs(d)>0.8 else "medium" if abs(d)>0.5 else "small" if abs(d)>0.2 else "negligible")
            st.markdown(f"""
            <div style='background:{C_WHITE};border-radius:6px;padding:14px 20px;
                        box-shadow:0 1px 3px rgba(0,0,0,0.06);
                        display:grid;grid-template-columns:repeat(4,1fr);gap:12px'>
              <div>
                <div class='kpi-label'>MAS mean (n={len(mas_v)})</div>
                <div style='font-size:1.1rem;font-weight:700;color:{C_MAS}'>{mas_v.mean():.4f}</div>
              </div>
              <div>
                <div class='kpi-label'>Single mean (n={len(sng_v)})</div>
                <div style='font-size:1.1rem;font-weight:700;color:{C_SINGLE}'>{sng_v.mean():.4f}</div>
              </div>
              <div>
                <div class='kpi-label'>t-test p-value</div>
                <div style='font-size:1.1rem;font-weight:700;color:{C_TEXT}'>{p_str}</div>
              </div>
              <div>
                <div class='kpi-label'>Mann-Whitney U p</div>
                <div style='font-size:1.1rem;font-weight:700;color:{C_TEXT}'>{up_str}</div>
              </div>
              <div>
                <div class='kpi-label'>Cohen's d</div>
                <div style='font-size:1.1rem;font-weight:700;color:{C_WIN if d>0 else C_LOSS}'>{d:+.3f} ({d_lbl})</div>
              </div>
              <div>
                <div class='kpi-label'>Delta (MAS - Single)</div>
                <div style='font-size:1.1rem;font-weight:700;color:{C_WIN if d>0 else C_LOSS}'>{mas_v.mean()-sng_v.mean():+.4f}</div>
              </div>
              <div>
                <div class='kpi-label'>MAS std</div>
                <div style='font-size:1.1rem;font-weight:700;color:{C_TEXT}'>{mas_v.std():.4f}</div>
              </div>
              <div>
                <div class='kpi-label'>Single std</div>
                <div style='font-size:1.1rem;font-weight:700;color:{C_TEXT}'>{sng_v.std():.4f}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FAILURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if "failure_mode" not in df.columns:
        st.info("failure_mode column not available in this run.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='section-title'>Failure Distribution by Architecture</div>", unsafe_allow_html=True)
            df2 = df.copy()
            df2["failure_mode"]  = df2["failure_mode"].fillna("error")
            df2["Architecture"]  = df2["mode"].map(MODE_MAP)
            counts = (df2.groupby(["Architecture","failure_mode"]).size()
                         .reset_index(name="count"))
            totals = counts.groupby("Architecture")["count"].transform("sum")
            counts["pct"] = 100 * counts["count"] / totals

            fig = px.bar(
                counts, x="Architecture", y="pct", color="failure_mode",
                color_discrete_map=FAIL_PALETTE,
                text=counts["pct"].round(0).astype(int).astype(str) + "%",
                labels={"pct": "Percentage of Runs (%)", "failure_mode": "Failure Mode"})
            fig.update_traces(textposition="inside", textfont_size=11, marker_line_width=0)
            _fmt(fig, height=360, barmode="stack",
                              legend=dict(title="Failure Mode", orientation="v"))
            st.plotly_chart(fig, width='stretch')

        with c2:
            st.markdown("<div class='section-title'>Failure Distribution by Difficulty</div>", unsafe_allow_html=True)
            if "difficulty" in df2.columns:
                counts2 = (df2.groupby(["difficulty","failure_mode"]).size()
                              .reset_index(name="count"))
                counts2["difficulty"] = pd.Categorical(counts2["difficulty"], DIFF_ORDER)
                counts2 = counts2.sort_values("difficulty")
                totals2 = counts2.groupby("difficulty")["count"].transform("sum")
                counts2["pct"] = 100 * counts2["count"] / totals2

                fig2 = px.bar(
                    counts2, x="difficulty", y="pct", color="failure_mode",
                    color_discrete_map=FAIL_PALETTE,
                    labels={"pct": "Percentage of Runs (%)", "difficulty": "Difficulty", "failure_mode": "Failure Mode"})
                fig2.update_traces(marker_line_width=0)
                _fmt(fig2, height=360, barmode="stack",
                                   legend=dict(title="Failure Mode"))
                st.plotly_chart(fig2, width='stretch')

        # ── Failure flow Sankey ───────────────────────────────────────────────
        st.markdown("<div class='section-title'>Failure Flow: Architecture — Failure Mode — Bug Pattern</div>", unsafe_allow_html=True)
        df2["pattern_simple"] = df2.get("pattern_matched", pd.Series(["none"]*len(df2))).fillna("none").apply(
            lambda x: x.split(",")[0].strip() if x and x != "none" else "none"
        )
        modes    = sorted(df2["Architecture"].unique())
        failures = sorted(df2["failure_mode"].unique())
        patterns = sorted(df2["pattern_simple"].unique())
        labels   = modes + failures + patterns
        idx      = {l: i for i, l in enumerate(labels)}
        src, tgt, val = [], [], []
        for (m, f), g in df2.groupby(["Architecture","failure_mode"]):
            src.append(idx[m]); tgt.append(idx[f]); val.append(len(g))
        for (f, p), g in df2.groupby(["failure_mode","pattern_simple"]):
            src.append(idx[f]); tgt.append(idx[p]); val.append(len(g))
        node_colors = [C_MAS if l=="MAS" else C_SINGLE if l=="Single"
                       else FAIL_PALETTE.get(l,"#BDC3C7") for l in labels]
        fig_sank = go.Figure(go.Sankey(
            node=dict(label=labels, color=node_colors, pad=18, thickness=22,
                      line=dict(width=0)),
            link=dict(source=src, target=tgt, value=val,
                      color="rgba(180,180,180,0.35)")))
        _fmt(fig_sank, height=380)
        fig_sank.update_xaxes(visible=False)
        fig_sank.update_yaxes(visible=False)
        st.plotly_chart(fig_sank, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REFORM JOURNEY
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    rh = pd.DataFrame(REFORM_DATA, columns=["Run","Reform","MAS","Single"])
    rh["Gap"] = rh["MAS"] - rh["Single"]

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("<div class='section-title'>patch_score Trajectory Across Reform Iterations</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rh["Run"], y=rh["MAS"], mode="lines+markers",
            name="MAS",
            line=dict(color=C_MAS, width=2.5),
            marker=dict(size=9, color=C_MAS, line=dict(color=C_WHITE, width=2)),
            hovertemplate="<b>%{x}</b><br>MAS: %{y:.3f}<br><i>%{text}</i><extra></extra>",
            text=rh["Reform"]))
        fig.add_trace(go.Scatter(
            x=rh["Run"], y=rh["Single"], mode="lines+markers",
            name="Single",
            line=dict(color=C_SINGLE, width=2, dash="dash"),
            marker=dict(size=8, color=C_SINGLE, line=dict(color=C_WHITE, width=2)),
            hovertemplate="<b>%{x}</b><br>Single: %{y:.3f}<extra></extra>"))
        # MAS win zone
        fig.add_vrect(x0="Run K", x1="Run M",
                      fillcolor=C_WIN, opacity=0.07,
                      annotation_text="MAS leads", annotation_position="top left",
                      annotation_font=dict(color=C_WIN, size=10))
        _fmt(fig, height=380,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown("<div class='section-title'>MAS vs Single Gap per Run</div>", unsafe_allow_html=True)
        bar_colors = [C_WIN if g > 0 else C_LOSS for g in rh["Gap"]]
        fig2 = go.Figure(go.Bar(
            x=rh["Run"], y=rh["Gap"],
            marker_color=bar_colors,
            marker_line_width=0,
            text=rh["Gap"].apply(lambda x: f"{x:+.3f}"),
            textposition="outside",
            textfont=dict(size=11),
            hovertemplate="<b>%{x}</b><br>Gap: %{y:+.3f}<br><i>%{customdata}</i><extra></extra>",
            customdata=rh["Reform"]))
        fig2.add_hline(y=0, line_color=C_TEXT, line_width=1)
        _fmt(fig2, height=380)
        fig2.update_yaxes(title_text="MAS - Single (patch_score)")
        fig2.update_xaxes(tickangle=-25)
        st.plotly_chart(fig2, width='stretch')

    # ── Reform impact summary ─────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Reform Impact Summary</div>", unsafe_allow_html=True)
    reform_rows = []
    for i, (run, reform, mas, sng) in enumerate(REFORM_DATA):
        gap  = mas - sng
        prev = REFORM_DATA[i-1][2] - REFORM_DATA[i-1][3] if i > 0 else None
        delta_gap = gap - prev if prev is not None else None
        reform_rows.append({
            "Run": run, "Reform Added": reform,
            "MAS": f"{mas:.3f}", "Single": f"{sng:.3f}",
            "Gap": f"{gap:+.3f}",
            "Gap Change": f"{delta_gap:+.3f}" if delta_gap is not None else "—",
            "Winner": "MAS" if gap > 0 else "Single",
        })

    reform_df = pd.DataFrame(reform_rows)

    def style_winner(val):
        if val == "MAS":     return f"color:{C_WIN};font-weight:700"
        if val == "Single":  return f"color:{C_LOSS};font-weight:600"
        return ""

    styled = (reform_df.style
              .map(style_winner, subset=["Winner"])
              .set_properties(**{"font-size": "12px", "text-align": "left"})
              .hide(axis="index"))
    st.dataframe(styled, width='stretch', height=310)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AGENT BEHAVIOUR
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2)

    # ── Debug improvement ─────────────────────────────────────────────────────
    with c1:
        st.markdown("<div class='section-title'>Debug Loop Improvement (patch_score delta)</div>", unsafe_allow_html=True)
        if "debug_improvement" in df.columns:
            df2 = df.dropna(subset=["debug_improvement"]).copy()
            df2["debug_improvement"] = pd.to_numeric(df2["debug_improvement"], errors="coerce")
            df2["Architecture"] = df2["mode"].map(MODE_MAP)
            fig = px.box(
                df2.dropna(subset=["debug_improvement","Architecture"]),
                x="Architecture", y="debug_improvement", color="Architecture",
                color_discrete_map={"MAS": C_MAS, "Single": C_SINGLE},
                points="all",
                labels={"debug_improvement": "patch_score delta (final - initial)", "Architecture": ""})
            fig.add_hline(y=0, line_dash="dot", line_color=C_SUB, line_width=1,
                          annotation_text="No change", annotation_font_size=10)
            fig.update_traces(jitter=0.3, marker=dict(size=5, opacity=0.55), line_width=1.5)
            _fmt(fig, height=340, showlegend=False)
            st.plotly_chart(fig, width='stretch')

    # ── LLM call count ────────────────────────────────────────────────────────
    with c2:
        st.markdown("<div class='section-title'>LLM Calls per Pipeline Run</div>", unsafe_allow_html=True)
        if "llm_calls" in df.columns:
            df2 = df.dropna(subset=["llm_calls"]).copy()
            df2["Architecture"] = df2["mode"].map(MODE_MAP)
            df2["llm_calls"] = pd.to_numeric(df2["llm_calls"], errors="coerce")
            fig = px.histogram(
                df2, x="llm_calls", color="Architecture", barmode="overlay",
                color_discrete_map={"MAS": C_MAS, "Single": C_SINGLE},
                opacity=0.72, nbins=10,
                labels={"llm_calls": "Number of LLM Calls", "count": "Run Count"})
            fig.update_traces(marker_line_width=0)
            _fmt(fig, height=340)
            st.plotly_chart(fig, width='stretch')

    # ── Convergence scatter ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Convergence: Initial vs Final patch_score by Difficulty</div>", unsafe_allow_html=True)
    st.caption("Points above the diagonal indicate the debug loop improved the patch. Points below indicate degradation.")
    if "initial_patch_score" in df.columns and "patch_score" in df.columns:
        df2 = df.dropna(subset=["initial_patch_score","patch_score"]).copy()
        for c in ["initial_patch_score","patch_score"]:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
        df2["Architecture"] = df2["mode"].map(MODE_MAP)
        df2 = df2.dropna(subset=["initial_patch_score","patch_score","Architecture"])

        diffs = df2["difficulty"].dropna().unique() if "difficulty" in df2.columns else ["all"]
        diffs = [d for d in DIFF_ORDER if d in diffs]
        if not diffs: diffs = ["all"]

        fig = make_subplots(rows=1, cols=len(diffs),
                            subplot_titles=[d.capitalize() for d in diffs],
                            shared_yaxes=True, shared_xaxes=True)

        for ci, diff in enumerate(diffs, 1):
            sub = df2[df2["difficulty"]==diff] if diff!="all" else df2
            for arch, color in [("MAS", C_MAS), ("Single", C_SINGLE)]:
                g = sub[sub["Architecture"]==arch]
                if g.empty: continue
                fig.add_trace(go.Scatter(
                    x=g["initial_patch_score"], y=g["patch_score"],
                    mode="markers",
                    name=arch, showlegend=(ci==1),
                    marker=dict(color=color, size=8, opacity=0.65,
                                line=dict(color=C_WHITE, width=1))), row=1, col=ci)
            rng = [0, 0.9]
            fig.add_trace(go.Scatter(
                x=rng, y=rng, mode="lines",
                line=dict(color=C_GRID, dash="dot", width=1.5),
                showlegend=(ci==1), name="No change"), row=1, col=ci)

        _fmt(fig, height=340,
                          legend=dict(orientation="h", yanchor="bottom", y=1.05))
        fig.update_xaxes(title_text="Initial patch_score", range=[-0.02, 0.95])
        fig.update_yaxes(title_text="Final patch_score",   range=[-0.02, 0.95])
        st.plotly_chart(fig, width='stretch')

    # ── Cost efficiency ───────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Cost Efficiency: patch_score per LLM Call</div>", unsafe_allow_html=True)
        if "llm_calls" in df.columns and "patch_score" in df.columns:
            df2 = df.copy()
            df2["efficiency"] = (pd.to_numeric(df2["patch_score"], errors="coerce") /
                                 pd.to_numeric(df2["llm_calls"], errors="coerce").replace(0, np.nan))
            df2["Architecture"] = df2["mode"].map(MODE_MAP)
            eff_grp = df2.groupby("Architecture")["efficiency"].mean().reset_index()
            fig = px.bar(
                eff_grp, x="Architecture", y="efficiency", color="Architecture",
                color_discrete_map={"MAS": C_MAS, "Single": C_SINGLE},
                text=eff_grp["efficiency"].round(4),
                labels={"efficiency": "patch_score / LLM call", "Architecture": ""})
            fig.update_traces(textposition="outside", marker_line_width=0, textfont_size=12)
            _fmt(fig, height=300, showlegend=False)
            fig.update_yaxes(range=[0, eff_grp["efficiency"].max() * 1.2])
            st.plotly_chart(fig, width='stretch')

    with c2:
        st.markdown("<div class='section-title'>Debug Iterations Distribution</div>", unsafe_allow_html=True)
        if "debug_iterations" in df.columns:
            df2 = df.dropna(subset=["debug_iterations"]).copy()
            df2["debug_iterations"] = pd.to_numeric(df2["debug_iterations"], errors="coerce")
            df2["Architecture"] = df2["mode"].map(MODE_MAP)
            fig = px.histogram(
                df2, x="debug_iterations", color="Architecture",
                barmode="overlay", opacity=0.72, nbins=8,
                color_discrete_map={"MAS": C_MAS, "Single": C_SINGLE},
                labels={"debug_iterations": "Number of Debug Iterations", "count": "Run Count"})
            fig.update_traces(marker_line_width=0)
            _fmt(fig, height=300)
            st.plotly_chart(fig, width='stretch')
