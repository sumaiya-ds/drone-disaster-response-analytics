from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.modeling import MODEL_RESULT_SCHEMA_VERSION, train_compare_models, validate_model_results

COLORS = {
    'background': '#F8F9F8',
    'surface': '#FFFFFF',
    'surface_alt': '#F4F6F5',
    'border': '#E8EBE9',
    'text': '#303634',
    'muted': '#7A817D',
    'green': '#56A078',
    'sage': '#BFDCCB',
    'blue': '#72A9C7',
    'amber': '#E0B72F',
    'red': '#E57375',
    'success': '#56A078',
    'cyan': '#72A9C7',
}

_STYLE = """
<style>
:root {
  color-scheme: light;
  --app-bg: #F8F9F8;
  --app-surface: #FFFFFF;
  --app-subtle: #F4F6F5;
  --app-border: #E8EBE9;
  --app-text: #303634;
  --app-muted: #7A817D;
  --app-green: #56A078;
}
.stApp { background: var(--app-bg); color: var(--app-text); }
.block-container { max-width: 1480px; padding: 1.7rem 2.4rem 3rem; }
header[data-testid="stHeader"] { background: rgba(248,249,248,.96); }
section[data-testid="stSidebar"] {
  background: #FFFFFF;
  border-right: 1px solid var(--app-border);
}
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--app-muted); }
[data-testid="stSidebarNav"] { padding: .5rem .3rem; }
[data-testid="stSidebarNav"] a { border-radius: 9px; margin: 3px 6px; color: #374151; }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  color: #347653;
  background: #EDF6F0;
  border-left: 2px solid #56A078;
}
[data-testid="stSidebarNav"] a:hover { background: #F4F6F5; }
[data-testid="stMetric"] {
  background: var(--app-surface);
  border: 1px solid var(--app-border);
  border-radius: 10px;
  padding: .85rem 1rem;
  box-shadow: 0 1px 2px rgba(37,41,37,.025);
}
[data-testid="stMetricLabel"] p { color: var(--app-muted); font-size: .79rem; }
[data-testid="stMetricValue"] { color: var(--app-text); }
[data-testid="stMetricDelta"] { font-size: .78rem; }
[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] { gap: 1rem; }
.app-brand { padding: .7rem .65rem 1rem; margin: 0 .4rem .8rem; border-bottom: 1px solid var(--app-border); }
.app-brand-rule { width: 28px; height: 3px; border-radius: 3px; background: #56A078; margin-bottom: .75rem; }
.app-brand-kicker { color: #41805F; font: 650 .62rem/1.3 sans-serif; letter-spacing: .13em; }
.app-brand-name { color: var(--app-text); font-size: .91rem; font-weight: 650; margin-top: .4rem; }
.app-brand-sub { color: var(--app-muted); font-size: .72rem; margin-top: .22rem; }
.page-eyebrow { color: #41805F; font-size: .65rem; font-weight: 650; letter-spacing: .12em; margin: .15rem 0 .35rem; }
.stApp h1 { color: var(--app-text); font-size: 1.7rem; font-weight: 640; letter-spacing: -.025em; line-height: 1.2; margin: 0; padding: 0; }
.page-description { color: #656B65; font-size: .88rem; margin-top: .42rem; max-width: 900px; line-height: 1.55; }
.page-status { color: #347653; background: #EDF6F0; border: 1px solid #D8E9DD; border-radius: 8px; padding: .38rem .65rem; font-size: .68rem; white-space: nowrap; }
.page-rule { border: 0; border-top: 1px solid var(--app-border); margin: 1rem 0 1.15rem; }
.kpi-card {
  min-height: 102px; height: 100%;
  background: color-mix(in srgb, var(--app-accent) 8%, #FFFFFF);
  border: 1px solid color-mix(in srgb, var(--app-accent) 15%, #E8EBE9);
  border-radius: 11px; padding: .9rem 1rem;
}
.kpi-label { color: var(--app-muted); font-size: .68rem; letter-spacing: .055em; }
.kpi-value { color: var(--app-text); font-size: 1.48rem; font-weight: 650; letter-spacing: -.025em; margin-top: .42rem; }
.kpi-note { color: #777D77; font-size: .7rem; margin-top: .22rem; }
.app-panel { background: #FFFFFF; border: 1px solid var(--app-border); border-radius: 10px; padding: .9rem 1rem; }
.app-panel-title { color: var(--app-text); font-weight: 620; font-size: .88rem; }
.app-panel-note { color: var(--app-muted); font-size: .76rem; margin-top: .3rem; line-height: 1.5; }
.section-title { color: var(--app-text); font-size: 1rem; font-weight: 640; margin: 1.2rem 0 .4rem; }
.section-note { color: var(--app-muted); font-size: .77rem; margin: 0 0 .7rem; }
.data-indicator { background: #FFFFFF; border: 1px solid var(--app-border); border-left: 3px solid var(--indicator-accent); border-radius: 8px; padding: .68rem .82rem; margin: .4rem 0; }
.data-indicator-title { color: var(--app-text); font-size: .8rem; font-weight: 620; }
.data-indicator-copy { color: var(--app-muted); font-size: .73rem; margin-top: .2rem; line-height: 1.45; }
.map-note { color: var(--app-muted); font-size: .75rem; margin: .25rem 0 .65rem; }
div[data-testid="stPlotlyChart"] {
  background: #FFFFFF;
  border: 1px solid var(--app-border);
  border-radius: 11px;
  padding: .35rem;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--app-border); border-radius: 9px; }
div[data-testid="stAlert"] { border-radius: 9px; }
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  border-color: var(--app-border);
  border-radius: 8px;
}
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
  border-radius: 8px;
  border-color: var(--app-border);
}
div[data-testid="stButton"] > button[kind="primary"] {
  background: #56A078;
  border-color: #56A078;
}
@media (max-width: 800px) {
  .block-container { padding: 1.15rem 1rem 2rem; }
  .stApp h1 { font-size: 1.45rem; }
  .kpi-card { min-height: 92px; }
}
</style>
"""


def apply_theme() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            """
            <div class="app-brand">
              <div class="app-brand-rule"></div>
              <div class="app-brand-kicker">DRONE RESPONSE INTELLIGENCE</div>
              <div class="app-brand-name">Drone Disaster Analysis</div>
              <div class="app-brand-sub">Assessment &amp; Rescue Analytics</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption('Historical and simulated observations · not a live telemetry feed')


def page_header(title: str, description: str, eyebrow: str = 'RESPONSE ANALYTICS', status: str | None = None) -> None:
    st.markdown(f'<div class="page-eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    title_col, status_col = st.columns([4, 1])
    with title_col:
        st.title(title)
    if status:
        with status_col:
            st.markdown('<div style="height:.42rem"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="page-status">{escape(status)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-description">{escape(description)}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="page-rule">', unsafe_allow_html=True)


def section_header(title: str, note: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{escape(note)}</div>', unsafe_allow_html=True)


def metric_cards(items: list[dict[str, Any]], columns: int | None = None) -> None:
    slots = st.columns(columns or len(items))
    for slot, item in zip(slots, items):
        accent = item.get('accent', COLORS['green'])
        label = escape(str(item['label']))
        value = escape(str(item['value']))
        note = escape(str(item.get('note', '')))
        slot.markdown(
            f"""
            <div class="kpi-card" style="--app-accent:{accent}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def alert_card(title: str, detail: str, severity: str = 'amber') -> None:
    accent = {
        'critical': COLORS['red'],
        'amber': COLORS['amber'],
        'green': COLORS['success'],
        'blue': COLORS['blue'],
    }.get(severity, COLORS['amber'])
    st.markdown(
        f"""
        <div class="data-indicator" style="--indicator-accent:{accent}">
          <div class="data-indicator-title">{escape(title)}</div>
          <div class="data-indicator-copy">{escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(figure: go.Figure, height: int | None = None) -> go.Figure:
    figure.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, Segoe UI, sans-serif', 'color': COLORS['text'], 'size': 11},
        title={'font': {'size': 14, 'color': COLORS['text']}, 'x': 0.02, 'xanchor': 'left'},
        margin={'l': 14, 'r': 14, 't': 48, 'b': 18},
        legend={'bgcolor': 'rgba(0,0,0,0)', 'font': {'color': COLORS['muted']}},
        hoverlabel={'bgcolor': '#FFFFFF', 'bordercolor': COLORS['border'], 'font': {'color': COLORS['text']}},
        height=height,
    )
    figure.update_xaxes(
        gridcolor='rgba(115,120,115,.12)',
        zerolinecolor='rgba(115,120,115,.16)',
        linecolor=COLORS['border'],
        tickfont={'color': COLORS['muted']},
        title_font={'color': COLORS['muted']},
        showline=False,
    )
    figure.update_yaxes(
        gridcolor='rgba(115,120,115,.12)',
        zerolinecolor='rgba(115,120,115,.16)',
        linecolor=COLORS['border'],
        tickfont={'color': COLORS['muted']},
        title_font={'color': COLORS['muted']},
        showline=False,
    )
    return figure


@st.cache_resource(show_spinner='Training and evaluating models on the loaded dataset...')
def _cached_model_evaluation(
    df: pd.DataFrame,
    target: str,
    result_schema_version: int,
) -> dict[str, Any]:
    if result_schema_version != MODEL_RESULT_SCHEMA_VERSION:
        raise ValueError(f'Unsupported model result schema version: {result_schema_version}.')
    return train_compare_models(df, target=target)


def evaluate_models(df: pd.DataFrame, target: str) -> dict[str, Any]:
    results = _cached_model_evaluation(df, target, MODEL_RESULT_SCHEMA_VERSION)
    return validate_model_results(results, target=target)


def clear_model_evaluation_cache() -> None:
    _cached_model_evaluation.clear()
