"""
Streamlit App - Customer Churn Prediction

Enterprise analytics dashboard for customer churn exploration and prediction.
Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.models.model_manager import ModelManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

APP_ICON = "📈"
PAGE_NAVIGATION = [
    ("🏠 Home", "home"),
    ("📈 Dashboard", "dashboard"),
    ("🔮 Predict Churn", "predict"),
    ("📚 Information", "information"),
]
PAGE_HERO_ICONS = {
    "home": "🏠",
    "dashboard": "📈",
    "predict": "🔮",
    "information": "📚",
}

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def text(value: object) -> str:
    return escape("") if value is None else escape(str(value))


def render_html(html: str) -> None:
    normalized_html = "\n".join(line.lstrip() for line in dedent(html).splitlines())
    st.markdown(normalized_html.strip(), unsafe_allow_html=True)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #050b14;
            --bg-1: #07111d;
            --panel: rgba(15, 25, 43, 0.72);
            --panel-strong: rgba(18, 31, 53, 0.88);
            --border: rgba(184, 206, 240, 0.14);
            --text: #eef5ff;
            --muted: #aabbd6;
            --danger: #ff7b86;
            --warning: #f4c26a;
            --success: #69d39d;
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.40);
            --shadow-soft: 0 16px 44px rgba(0, 0, 0, 0.24);
            --radius-lg: 20px;
        }

        html, body, [class*="css"] {
            font-family: "Aptos", "Segoe UI Variable", "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 12%, rgba(110, 202, 255, 0.10), transparent 24%),
                radial-gradient(circle at 84% 16%, rgba(111, 231, 216, 0.10), transparent 18%),
                radial-gradient(circle at 82% 78%, rgba(194, 141, 255, 0.08), transparent 22%),
                linear-gradient(180deg, var(--bg-0), var(--bg-1) 36%, #071423 100%);
            color: var(--text);
        }

        .stApp header {
            background: linear-gradient(90deg, rgba(5, 11, 20, 0.65), rgba(7, 17, 29, 0.18));
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(10px);
        }

        section[data-testid="stMain"] {
            background: transparent;
            min-width: 0 !important;
        }

        section[data-testid="stMain"] > div {
            background: transparent;
        }

        .block-container {
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
            padding-top: 1.1rem;
            padding-bottom: 2.2rem;
        }

        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top right, rgba(110, 202, 255, 0.16), transparent 22%),
                linear-gradient(180deg, rgba(6, 11, 20, 0.98), rgba(8, 14, 25, 0.94));
            border-right: 1px solid rgba(255, 255, 255, 0.04);
            box-shadow: 18px 0 44px rgba(0, 0, 0, 0.26);
            backdrop-filter: blur(16px);
            transition: width 180ms ease, min-width 180ms ease, max-width 180ms ease, flex-basis 180ms ease;
        }

        section[data-testid="stSidebar"] > div {
            padding: 1rem 0.95rem 1.15rem;
        }

        .stSidebar [data-testid="stRadio"] {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 18px;
            padding: 0.6rem;
            box-shadow: var(--shadow-soft);
        }

        .stSidebar [data-testid="stRadio"] div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
        }

        .stSidebar [data-testid="stRadio"] label {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            min-height: 2.95rem;
            padding: 0.68rem 0.72rem;
            border-radius: 14px;
            border: 1px solid transparent;
            background: rgba(255, 255, 255, 0.01);
            color: var(--muted) !important;
            font-weight: 700;
            transition: transform 160ms ease, background 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
        }

        .stSidebar [data-testid="stRadio"] label:hover {
            transform: translateX(2px);
            background: rgba(110, 202, 255, 0.07);
            border-color: rgba(110, 202, 255, 0.10);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
        }

        .stSidebar [data-testid="stRadio"] label[data-checked="true"] {
            color: var(--text) !important;
            background: linear-gradient(135deg, rgba(110, 202, 255, 0.16), rgba(111, 231, 216, 0.08));
            border-color: rgba(110, 202, 255, 0.24);
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.28);
        }

        .stSidebar [data-testid="stRadio"] label p,
        .stSidebar [data-testid="stRadio"] label span {
            color: inherit !important;
            font-weight: 700 !important;
        }

        .sidebar-shell,
        .hero-card,
        .metric-card,
        .feature-card,
        .info-card,
        .chart-card,
        .prediction-card,
        .sidebar-stat,
        .panel-card,
        .architecture-node {
            position: relative;
            overflow: hidden;
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            background:
                radial-gradient(circle at top right, rgba(110, 202, 255, 0.12), transparent 28%),
                linear-gradient(180deg, var(--panel-strong), var(--panel));
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(16px);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, background 180ms ease;
        }

        .sidebar-shell:hover,
        .hero-card:hover,
        .metric-card:hover,
        .feature-card:hover,
        .info-card:hover,
        .chart-card:hover,
        .prediction-card:hover,
        .sidebar-stat:hover,
        .panel-card:hover,
        .architecture-node:hover {
            transform: translateY(-3px);
            border-color: rgba(110, 202, 255, 0.26);
            box-shadow: 0 22px 54px rgba(0, 0, 0, 0.34);
        }

        .sidebar-shell {
            padding: 0.95rem;
            margin-bottom: 0.9rem;
            background: linear-gradient(180deg, rgba(16, 29, 47, 0.70), rgba(8, 13, 23, 0.90));
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .sidebar-logo {
            width: 2.75rem;
            height: 2.75rem;
            border-radius: 14px;
            display: grid;
            place-items: center;
            font-size: 1.25rem;
            background: linear-gradient(135deg, rgba(110, 202, 255, 0.24), rgba(111, 231, 216, 0.16));
            border: 1px solid rgba(184, 206, 240, 0.16);
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
        }

        .sidebar-brand h3 {
            margin: 0;
            font-size: 1rem;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }

        .sidebar-brand p {
            margin: 0.2rem 0 0;
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.35;
        }

        .sidebar-stat,
        .metric-card,
        .feature-card,
        .info-card,
        .prediction-card,
        .panel-card,
        .architecture-node {
            padding: 1rem 1.05rem;
        }

        .sidebar-stat {
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
        }

        .stSidebar .kpi-grid {
            grid-template-columns: 1fr;
            gap: 0.65rem;
        }

        .stSidebar .metric-card {
            min-height: auto;
            padding: 0.85rem 0.9rem;
        }

        .stSidebar .metric-card .value {
            font-size: 0.98rem;
            line-height: 1.2;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
            gap: 0.72rem;
        }

        .tight-metric-shell {
            border: 1px solid rgba(184, 206, 240, 0.14);
            border-radius: 20px;
            padding: 0.75rem;
            background: rgba(255, 255, 255, 0.02);
            box-shadow: var(--shadow-soft);
        }

        .tight-metric-shell .kpi-grid {
            gap: 0.72rem;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
        }

        .two-up {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
        }

        .hero-card {
            padding: 1.9rem;
            background:
                radial-gradient(circle at top right, rgba(111, 231, 216, 0.18), transparent 25%),
                radial-gradient(circle at left center, rgba(142, 168, 255, 0.18), transparent 30%),
                linear-gradient(135deg, rgba(16, 29, 49, 0.96), rgba(9, 16, 28, 0.88));
            margin-bottom: 1.35rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.7fr) minmax(240px, 0.7fr);
            gap: 1.1rem;
            align-items: stretch;
        }

        .hero-content {
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }

        .hero-side {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .hero-side h3 {
            margin: 0;
            font-size: 1.25rem;
            line-height: 1.12;
            letter-spacing: -0.03em;
        }

        .hero-side p {
            margin: 0.85rem 0 0;
            color: #c2d3ea;
            line-height: 1.55;
        }

        .hero-kicker,
        .section-badge,
        .status-pill,
        .soft-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.38rem 0.74rem;
            border-radius: 999px;
            border: 1px solid rgba(184, 206, 240, 0.16);
            color: #dce8fa;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2.0rem, 2.9vw, 3.15rem);
            line-height: 1.0;
            letter-spacing: -0.04em;
            font-weight: 900;
        }

        .hero-subtitle {
            margin: 0;
            max-width: 68ch;
            color: #c2d3ea;
            font-size: 0.98rem;
            line-height: 1.6;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.05rem;
        }

        .hero-side {
            border-radius: 24px;
            border: 1px solid rgba(184, 206, 240, 0.14);
            background:
                radial-gradient(circle at top right, rgba(110, 202, 255, 0.14), transparent 32%),
                linear-gradient(180deg, rgba(19, 34, 56, 0.90), rgba(10, 18, 31, 0.86));
            padding: 1.25rem;
            box-shadow: var(--shadow-soft);
        }

        .hero-side h3 {
            margin-bottom: 0.6rem;
        }

        .hero-side p {
            margin: 0;
        }

        .hero-side h3,
        .chart-title,
        .section-block h2,
        .architecture-node .node-title {
            margin: 0;
            letter-spacing: -0.02em;
        }

        .hero-side p,
        .section-block p,
        .chart-copy,
        .card-note,
        .metric-card .note,
        .feature-card .note,
        .info-card .note,
        .sidebar-stat .note,
        .architecture-node .node-copy {
            color: var(--muted);
            line-height: 1.6;
        }

        .metric-card {
            min-height: 104px;
            padding: 0.8rem 0.95rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }

        .label {
            color: #d9e6fa;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .value {
            color: var(--text);
            font-weight: 900;
            letter-spacing: -0.03em;
        }

        .metric-card .value {
            font-size: clamp(1.55rem, 2.2vw, 2.45rem);
            line-height: 1.05;
            margin-top: 0.22rem;
        }

        .feature-card,
        .info-card {
            min-height: 160px;
        }

        .feature-card .value,
        .info-card .value,
        .sidebar-stat .value {
            font-size: 1rem;
            line-height: 1.5;
            margin-top: 0.6rem;
        }

        .metric-card .note {
            margin-top: 0.22rem;
            color: #c1d2ea;
            font-size: 0.88rem;
        }

        .feature-card .value,
        .info-card .value {
            margin-top: 0.55rem;
        }

        .section-block {
            margin-top: 0.4rem;
            margin-bottom: 1.25rem;
        }

        .section-block h2 {
            margin-top: 0.15rem;
            font-size: 1.55rem;
            line-height: 1.1;
        }

        .section-block p {
            margin: 0.35rem 0 0;
        }

        .prediction-form-tight {
            margin-top: 0.15rem;
        }

        .prediction-form-tight [data-testid="column"] {
            gap: 0.35rem;
        }

        .prediction-form-tight label,
        .prediction-form-tight .stSelectbox,
        .prediction-form-tight .stNumberInput {
            margin-bottom: 0.1rem;
        }

        .prediction-form-tight [data-baseweb="select"] > div,
        .prediction-form-tight [data-baseweb="input"] {
            min-height: 2.45rem;
        }

        /* Force the main metric block container to not stretch */
        [data-testid="stMetric"] {
            background-color: #f8f9fa; /* adjust color if needed */
            border: 1px solid #e6e9ef;
            padding: 10px 15px !important;
            border-radius: 8px;
            max-height: 110px !important; /* Forces it to stay short */
            display: flex;
            flex-direction: column;
            justify-content: center !important;
        }

        /* Minimize spacing between the metric label and value */
        [data-testid="stMetricLabel"] {
            margin-bottom: -5px !important;
            line-height: 1.2 !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            line-height: 1.1 !important;
        }

        .chart-card {
            padding: 1rem 1rem 0.65rem;
        }

        .chart-title {
            font-size: 1rem;
            font-weight: 800;
        }

        .chart-copy {
            margin: 0.4rem 0 0.75rem;
            font-size: 0.93rem;
        }

        div[data-testid="stPlotlyChart"],
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(184, 206, 240, 0.12);
            background: rgba(255, 255, 255, 0.02);
            box-shadow: var(--shadow-soft);
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(142, 168, 255, 0.24);
            background: linear-gradient(135deg, rgba(142, 168, 255, 0.24), rgba(111, 231, 216, 0.16));
            color: white;
            font-weight: 800;
            padding: 0.7rem 1rem;
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease, filter 160ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 30px rgba(0, 0, 0, 0.24);
            border-color: rgba(111, 231, 216, 0.40);
            filter: brightness(1.04);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] {
            background: rgba(255, 255, 255, 0.04) !important;
            border-color: rgba(184, 206, 240, 0.14) !important;
            border-radius: 14px !important;
        }

        .stNumberInput input,
        .stTextInput input {
            color: var(--text) !important;
        }

        .status-high {
            color: var(--danger);
            border-color: rgba(255, 123, 134, 0.26);
            background: rgba(255, 123, 134, 0.10);
        }

        .status-medium {
            color: var(--warning);
            border-color: rgba(244, 194, 106, 0.26);
            background: rgba(244, 194, 106, 0.10);
        }

        .status-low {
            color: var(--success);
            border-color: rgba(105, 211, 157, 0.26);
            background: rgba(105, 211, 157, 0.10);
        }

        .architecture-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.9rem;
        }

        .architecture-node {
            min-height: 132px;
        }

        .architecture-node .node-kicker {
            color: #dce8fa;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .architecture-node .node-title {
            font-size: 1rem;
            font-weight: 800;
        }

        .footer-note {
            color: var(--muted);
            text-align: center;
            padding-top: 0.5rem;
            font-size: 0.92rem;
        }

        .section-shell {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.35rem;
        }

        .section-shell h2 {
            margin: 0.15rem 0 0;
        }

        @media (max-width: 1200px) {
            .hero-grid,
            .two-up {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 760px) {
            section[data-testid="stSidebar"] {
                width: 100% !important;
                min-width: 100% !important;
                max-width: 100% !important;
            }

            .block-container {
                padding-top: 0.9rem;
                padding-bottom: 1.25rem;
            }

            .kpi-grid,
            .feature-grid,
            .architecture-grid {
                grid-template-columns: 1fr;
            }

            .hero-card,
            .metric-card,
            .feature-card,
            .info-card,
            .chart-card,
            .prediction-card,
            .panel-card,
            .architecture-node,
            .sidebar-shell,
            .sidebar-stat {
                padding: 0.95rem;
            }

            .hero-title {
                font-size: 1.9rem;
            }

            .hero-grid {
                gap: 1rem;
            }

            .hero-content {
                gap: 0.8rem;
            }

            .section-block h2 {
                font-size: 1.3rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str | None = None, badge: str | None = None) -> None:
    badge_html = f'<div class="section-badge">{text(badge)}</div>' if badge else ""
    subtitle_html = f'<p>{text(subtitle)}</p>' if subtitle else ""
    render_html(
        f'''
        <div class="section-block">
            <div class="section-shell">
                <div>
                    {badge_html}
                    <h2>{text(title)}</h2>
                    {subtitle_html}
                </div>
            </div>
        </div>
        '''
    )


def render_hero(title: str, subtitle: str, chips: list[str], icon: str = APP_ICON) -> None:
    chips_html = ''.join(f'<span class="soft-badge">{text(chip)}</span>' for chip in chips)
    render_html(
        f'''
        <div class="hero-card">
            <div class="hero-grid">
                <div class="hero-content">
                    <div class="hero-kicker">Customer Retention Intelligence</div>
                    <div class="hero-title">{text(icon)} {text(title)}</div>
                    <p class="hero-subtitle">{text(subtitle)}</p>
                    <div class="badge-row">{chips_html}</div>
                </div>
                <div class="hero-side">
                    <h3>Executive Snapshot</h3>
                    <p>Built for enterprise demos with strong visual hierarchy, compact navigation, and fast churn scoring workflows.</p>
                    <div style="margin-top:0.9rem;display:grid;gap:0.6rem;">
                        <div class="status-pill">Production ready</div>
                        <div class="status-pill">Interactive analytics</div>
                        <div class="status-pill">Responsive layout</div>
                    </div>
                </div>
            </div>
        </div>
        '''
    )


def render_metrics(items: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, note in items:
        cards.append(
            f'''
            <div class="metric-card">
                <div class="label">{text(label)}</div>
                <div class="value">{text(value)}</div>
                <div class="note">{text(note)}</div>
            </div>
            '''
        )

    with st.container():
        st.markdown(
            """
            <style>
            .tight-metric-shell {
                border: 1px solid rgba(184, 206, 240, 0.14);
                border-radius: 20px;
                padding: 0.75rem;
                background: rgba(255, 255, 255, 0.02);
                box-shadow: var(--shadow-soft);
            }
            .tight-metric-shell .kpi-grid {
                gap: 0.72rem;
            }
            .tight-metric-shell .metric-card {
                min-height: 104px;
                padding: 0.8rem 0.95rem;
                justify-content: center;
                align-items: center;
                text-align: center;
            }
            .tight-metric-shell .metric-card .value {
                margin-top: 0.22rem;
            }
            .tight-metric-shell .metric-card .note {
                margin-top: 0.22rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        render_html(f'<div class="tight-metric-shell"><div class="kpi-grid">{"".join(cards)}</div></div>')


def render_feature_cards(items: list[tuple[str, str]]) -> None:
    cards = []
    for title, body in items:
        cards.append(
            f'''
            <div class="feature-card">
                <div class="label">{text(title)}</div>
                <div class="value">{text(body)}</div>
            </div>
            '''
        )
    render_html(f'<div class="feature-grid">{"".join(cards)}</div>')


def render_info_cards(items: list[tuple[str, str, str | None]]) -> None:
    cards = []
    for title, body, note in items:
        note_html = f'<div class="note">{text(note)}</div>' if note else ""
        cards.append(
            f'''
            <div class="info-card">
                <div class="label">{text(title)}</div>
                <div class="value">{text(body)}</div>
                {note_html}
            </div>
            '''
        )
    render_html(f'<div class="feature-grid">{"".join(cards)}</div>')


def chart_layout(fig: go.Figure, *, height: int = 360, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=18, b=12),
        font=dict(color="#e9f2ff", family='Aptos, Segoe UI Variable, Segoe UI, sans-serif'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0.02),
    )
    return fig


@st.cache_resource
def load_trained_model():
    try:
        model_path = PROJECT_ROOT / "models" / "churn_model_best.pkl"
        if not model_path.exists():
            st.error("Model not found. Please run `python main.py` first to train the model.")
            return None, None, None, None
        return ModelManager.load_model(str(model_path))
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return None, None, None, None


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    data_path = Path(config.get("data.raw_path", "data/raw/churn.csv"))
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    # Try a set of sensible candidate locations (config path, repo raw, processed)
    candidates = [
        data_path,
        PROJECT_ROOT / "data" / "raw" / "churn.csv",
        PROJECT_ROOT / "data" / "processed" / "churn_processed.csv",
    ]

    for candidate in candidates:
        try:
            if candidate.exists():
                return pd.read_csv(candidate)
        except Exception:
            # ignore parsing/open errors here and try next candidate
            continue

    # Fallback: try loading from a GitHub raw URL configured in config.yaml (useful for deployed hosts)
    github_raw = None
    try:
        cfg_streamlit = config.get("streamlit", {}) if isinstance(config, dict) else {}
        github_raw = cfg_streamlit.get("github_raw_url") if cfg_streamlit else None
    except Exception:
        github_raw = None

    if github_raw:
        try:
            df = pd.read_csv(github_raw)
            return df
        except Exception:
            # ignore network/parse errors and continue to return None
            pass

    # If all attempts failed, return None so the UI can offer an upload fallback.
    return None


def risk_state(probability: float) -> tuple[str, str]:
    if probability >= 0.7:
        return "HIGH", "status-high"
    if probability >= 0.4:
        return "MEDIUM", "status-medium"
    return "LOW", "status-low"


inject_styles()

model, scaler, preprocessor, metadata = load_trained_model()
metrics = metadata.get("metrics", {}) if metadata else {}

with st.sidebar:
    render_html(
        '''
        <div class="sidebar-shell">
            <div class="sidebar-brand">
                <div class="sidebar-logo">📈</div>
                <div>
                    <h3>Churn AI Console</h3>
                    <p>Enterprise retention analytics</p>
                </div>
            </div>
        </div>
        '''
    )

    page = st.radio(
        "Navigation",
        [label for label, _ in PAGE_NAVIGATION],
        label_visibility="collapsed",
    )

    page_key = next(key for label, key in PAGE_NAVIGATION if label == page)

    if metadata:
        model_name = metadata.get("model_type", "AdaBoost")
        render_html(
            f'''
            <div class="sidebar-stat">
                <div class="label">Model</div>
                <div class="value">{model_name}</div>
                <div class="note">Production artifact loaded from models/churn_model_best.pkl</div>
            </div>
            '''
        )
        render_metrics(
            [
                ("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.4f}", "Primary ranking metric"),
                ("Accuracy", f"{metrics.get('Accuracy', 0):.4f}", "Overall correctness"),
                ("Recall", f"{metrics.get('Recall', 0):.4f}", "Churn capture rate"),
            ]
        )

if page == "🏠 Home":
    render_hero(
        "Customer Churn Prediction",
        "Enterprise AI/ML analytics platform for monitoring churn risk, customer behavior, retention opportunities, and predictive insights.",
        [metadata.get("model_type", "AdaBoost"), "Production", "AI Analytics", "Customer Retention"],
        PAGE_HERO_ICONS[page_key],
    )

    sample_df = load_sample_data()
    if sample_df is None:
        uploaded = st.file_uploader("Upload sample churn.csv (optional)", type=["csv"])
        if uploaded is not None:
            try:
                sample_df = pd.read_csv(uploaded)
                churn_rate = float((sample_df["Churn"] == "Yes").mean() * 100)
            except Exception:
                st.error("Uploaded file could not be read as a CSV. Please upload a valid churn.csv file.")
                sample_df = None
                churn_rate = 0.0
        else:
            sample_df = None
            churn_rate = 0.0
    else:
        churn_rate = float((sample_df["Churn"] == "Yes").mean() * 100)

    render_section_header("KPI Snapshot", "Core metrics for the current churn model.", "Executive Summary")
    render_metrics(
        [
            ("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.4f}", "Primary ranking metric"),
            ("Accuracy", f"{metrics.get('Accuracy', 0):.4f}", "Correct classification rate"),
            ("Precision", f"{metrics.get('Precision', 0):.4f}", "Positive prediction quality"),
            ("Recall", f"{metrics.get('Recall', 0):.4f}", "Churn capture rate"),
            ("Customer Risk", f"{churn_rate:.1f}%", "Observed churn exposure"),
        ]
    )

    st.write("")
    render_section_header("Executive Overview", "Enterprise feature cards for operational storytelling and retention strategy.")
    render_feature_cards(
        [
            ("Detect Churn Risk", "Prioritize high-risk customers before retention windows close."),
            ("Improve Retention", "Focus offers and interventions where the business impact is highest."),
            ("Analyze Customer Behavior", "Surface signal patterns across tenure, billing, and service usage."),
            ("Predict Customer Loss", "Support faster, model-backed decisions with measurable confidence."),
        ]
    )

    st.write("")
    render_section_header("Visual Analytics", "Charts are placed in premium containers so they never float on the page background.")

    if sample_df is not None:
        left, right = st.columns(2)
        churn_counts = sample_df["Churn"].value_counts()

        churn_fig = go.Figure(
            data=[
                go.Pie(
                    labels=["No Churn", "Churn"],
                    values=[churn_counts.get("No", 0), churn_counts.get("Yes", 0)],
                    hole=0.6,
                    sort=False,
                    direction="clockwise",
                    marker=dict(colors=["#69d39d", "#ff7b86"], line=dict(color="rgba(255,255,255,0.10)", width=1)),
                )
            ]
        )
        chart_layout(churn_fig, height=340)
        with left:
            st.markdown('<div class="chart-card"><div class="chart-title">Churn Distribution</div><div class="chart-copy">Share of retained versus churned customers.</div></div>', unsafe_allow_html=True)
            st.plotly_chart(churn_fig, width="stretch", config={"displayModeBar": False})

        contract_rate = sample_df.groupby("Contract")["Churn"].apply(lambda series: (series == "Yes").mean() * 100).sort_values(ascending=False)
        contract_fig = go.Figure(data=[go.Bar(x=contract_rate.index, y=contract_rate.values, marker_color=["#8ea8ff", "#69d39d", "#f4c26a"])])
        contract_fig.update_yaxes(gridcolor="rgba(184,206,240,0.14)", title_text="Churn rate %")
        chart_layout(contract_fig, height=340, showlegend=False)
        with right:
            st.markdown('<div class="chart-card"><div class="chart-title">Risk by Contract Type</div><div class="chart-copy">Churn exposure by subscription structure.</div></div>', unsafe_allow_html=True)
            st.plotly_chart(contract_fig, width="stretch", config={"displayModeBar": False})

        left2, right2 = st.columns(2)
        tenure_bins = pd.cut(sample_df["tenure"], bins=[0, 12, 24, 48, 72], include_lowest=True)
        tenure_trend = (
            sample_df.assign(TenureBand=tenure_bins)
            .groupby("TenureBand", observed=False)["Churn"]
            .apply(lambda series: (series == "Yes").mean() * 100)
            .reset_index(name="ChurnRate")
        )
        trend_fig = go.Figure(data=[go.Scatter(x=[str(x) for x in tenure_trend["TenureBand"]], y=tenure_trend["ChurnRate"], mode="lines+markers", line=dict(color="#6fe7d8", width=3), marker=dict(size=8))])
        trend_fig.update_yaxes(gridcolor="rgba(184,206,240,0.14)", title_text="Churn rate %")
        chart_layout(trend_fig, height=320, showlegend=False)
        with left2:
            st.markdown('<div class="chart-card"><div class="chart-title">Prediction Trend</div><div class="chart-copy">Churn intensity across customer tenure bands.</div></div>', unsafe_allow_html=True)
            st.plotly_chart(trend_fig, width="stretch", config={"displayModeBar": False})

        monthly_fig = go.Figure()
        monthly_fig.add_trace(go.Box(y=sample_df[sample_df["Churn"] == "No"]["MonthlyCharges"], name="Retained", marker_color="#69d39d", boxmean=True))
        monthly_fig.add_trace(go.Box(y=sample_df[sample_df["Churn"] == "Yes"]["MonthlyCharges"], name="Churn", marker_color="#ff7b86", boxmean=True))
        monthly_fig.update_yaxes(gridcolor="rgba(184,206,240,0.14)")
        chart_layout(monthly_fig, height=320)
        with right2:
            st.markdown('<div class="chart-card"><div class="chart-title">Segmentation Risk</div><div class="chart-copy">Monthly charge distribution across churn outcomes.</div></div>', unsafe_allow_html=True)
            st.plotly_chart(monthly_fig, width="stretch", config={"displayModeBar": False})
    else:
        st.info("Sample data is unavailable, so analytics charts could not be rendered.")

    st.write("")
    render_section_header("Model Information", "Structured summary of the saved artifact and deployment posture.")
    left_model, right_model = st.columns([1.25, 0.75])
    with left_model:
        render_info_cards(
            [
                ("Model Type", metadata.get("model_type", "N/A") if metadata else "N/A", None),
                ("Dataset Status", "Churn dataset loaded and aligned to the saved artifact.", None),
                ("Deployment Readiness", "Ready for Streamlit demo and local prediction workflows.", None),
                ("Prediction Workflow", "Input customer features, score risk, and trigger retention actions.", None),
            ]
        )
    with right_model:
        render_info_cards(
            [
                ("Evaluation Summary", metadata.get("model_type", "AdaBoost"), f"Accuracy {metrics.get('Accuracy', 0):.4f} • F1 {metrics.get('F1 Score', 0):.4f}"),
                ("Primary Metric", f"ROC-AUC {metrics.get('ROC-AUC', 0):.4f}", f"Recall {metrics.get('Recall', 0):.4f}"),
            ]
        )

    st.write("")
    render_section_header("Workflow", "End-to-end operating flow for the churn analytics experience.")
    render_info_cards(
        [
            ("1. Upload or input data", "Bring in a customer record or use the sample dataset for analysis.", None),
            ("2. Process attributes", "Standardize the customer profile and align it to the saved feature schema.", None),
            ("3. Predict churn", "Generate churn probability and associated risk classification.", None),
            ("4. Analyze results", "Review metrics, charts, and segment-level insights.", None),
        ]
    )

elif page == "📈 Dashboard":
    render_hero(
        "Dashboard & Statistics",
        "A clean command center for dataset health, churn distribution, and business context.",
        ["Dataset summary", "Visual analytics", "One-click loading"],
        PAGE_HERO_ICONS[page_key],
    )

    render_section_header("Dashboard Summary", "The dataset loads automatically so the key summary and charts appear immediately.")

    try:
        with st.spinner("Loading dataset and preparing summary..."):
            df = load_sample_data()
            # if not available, allow uploader fallback
            if df is None:
                uploaded = st.file_uploader("Upload sample churn.csv (optional)", type=["csv"])
                if uploaded is not None:
                    try:
                        df = pd.read_csv(uploaded)
                        st.session_state.sample_data = df
                    except Exception:
                        st.error("Uploaded file could not be read as a CSV. Please upload a valid churn.csv file.")
                        df = None
                else:
                    df = None
            else:
                st.session_state.sample_data = df

        if df is None:
            raise FileNotFoundError("No sample data available and no upload provided.")

        churn_count = int((df["Churn"] == "Yes").sum())
        churn_rate = churn_count / len(df) * 100
        average_tenure = float(df["tenure"].mean())

        render_metrics(
            [
                ("Total customers", f"{len(df):,}", "Rows loaded from churn.csv"),
                ("Churned", f"{churn_count:,}", "Observed churn outcomes"),
                ("Churn rate", f"{churn_rate:.1f}%", "Share of churned users"),
                ("Average tenure", f"{average_tenure:.0f} months", "Mean customer lifetime"),
                ("Model family", metadata.get("model_type", "AdaBoost") if metadata else "AdaBoost", "Current production artifact"),
            ]
        )

        st.write("")
        left, right = st.columns(2)

        churn_counts = df["Churn"].value_counts()
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["No Churn", "Churn"],
                    values=[churn_counts.get("No", 0), churn_counts.get("Yes", 0)],
                    hole=0.6,
                    sort=False,
                    direction="clockwise",
                    marker=dict(colors=["#69d39d", "#ff7b86"], line=dict(color="rgba(255,255,255,0.10)", width=1)),
                )
            ]
        )
        chart_layout(fig, height=350)
        with left:
            st.markdown('<div class="chart-card"><div class="chart-title">Churn Distribution</div><div class="chart-copy">Share of churned versus retained customers.</div></div>', unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        fig = go.Figure()
        fig.add_trace(go.Box(y=df[df["Churn"] == "No"]["tenure"], name="No Churn", marker_color="#69d39d", boxmean=True))
        fig.add_trace(go.Box(y=df[df["Churn"] == "Yes"]["tenure"], name="Churn", marker_color="#ff7b86", boxmean=True))
        fig.update_yaxes(gridcolor="rgba(184,206,240,0.14)", title_text="Months")
        chart_layout(fig, height=350)
        with right:
            st.markdown('<div class="chart-card"><div class="chart-title">Tenure vs Churn</div><div class="chart-copy">Tenure profile separated by churn outcome.</div></div>', unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.write("")
        render_info_cards(
            [
                ("Interpretation", "Higher churn density among early-tenure customers signals a retention window worth prioritizing.", None),
                ("Billing lens", "Monthly billing pressure and contract type are the strongest business-facing signals to watch.", None),
                ("Usage context", "Use the dashboard to explain customer segments before moving to manual prediction.", None),
            ]
        )
    except Exception as error:
        st.error(f"Could not load sample data: {error}")

elif page == "🔮 Predict Churn":
    render_hero(
        "Predict Customer Churn",
        "Enter customer attributes and receive an instant churn probability with a clear recommended action.",
        ["Probability scoring", "Risk labeling", "Action-oriented output"],
        PAGE_HERO_ICONS[page_key],
    )

    if model is None:
        st.error("Model not loaded. Please train the model first.")
    else:
        render_section_header("Prediction Workspace", "The form below is aligned with the saved training schema so predictions are stable and fast.")

        with st.form("prediction_form"):
            form_shell = st.container()
            with form_shell:
                st.markdown('<div class="prediction-form-tight">', unsafe_allow_html=True)
            left_form, right_form = st.columns(2, gap="small")

            with left_form:
                st.markdown('<div class="panel-card"><div class="section-badge">Customer Information</div><p class="chart-copy" style="margin-top:0.55rem;">Demographic and household attributes.</p></div>', unsafe_allow_html=True)
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                gender = st.selectbox("Gender", ["Male", "Female"])
                partner = st.selectbox("Has Partner", ["Yes", "No"])
                dependents = st.selectbox("Has Dependents", ["Yes", "No"])

                st.markdown('<div class="panel-card"><div class="section-badge">Service Usage</div><p class="chart-copy" style="margin-top:0.55rem;">Connectivity and support profile.</p></div>', unsafe_allow_html=True)
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

            with right_form:
                st.markdown('<div class="panel-card"><div class="section-badge">Account Information</div><p class="chart-copy" style="margin-top:0.55rem;">Tenure and billing signals.</p></div>', unsafe_allow_html=True)
                tenure = st.number_input("Tenure (months)", 0, 72, 24)
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
                total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)

                st.markdown('<div class="panel-card"><div class="section-badge">Additional Services</div><p class="chart-copy" style="margin-top:0.55rem;">Support and billing configuration.</p></div>', unsafe_allow_html=True)
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

            submitted = st.form_submit_button("🔮 Predict Churn")

            st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            customer_data = pd.DataFrame(
                {
                    "tenure": [tenure],
                    "MonthlyCharges": [monthly_charges],
                    "TotalCharges": [total_charges],
                    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
                    "gender_Male": [1 if gender == "Male" else 0],
                    "Partner_Yes": [1 if partner == "Yes" else 0],
                    "Dependents_Yes": [1 if dependents == "Yes" else 0],
                    "PhoneService_Yes": [1 if phone_service == "Yes" else 0],
                    "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
                    "InternetService_No": [1 if internet_service == "No" else 0],
                    "OnlineSecurity_No internet service": [1 if online_security == "No internet service" else 0],
                    "OnlineSecurity_Yes": [1 if online_security == "Yes" else 0],
                    "TechSupport_No internet service": [1 if tech_support == "No internet service" else 0],
                    "TechSupport_Yes": [1 if tech_support == "Yes" else 0],
                    "StreamingTV_No internet service": [1 if streaming_tv == "No internet service" else 0],
                    "StreamingTV_Yes": [1 if streaming_tv == "Yes" else 0],
                    "Contract_One year": [1 if contract == "One year" else 0],
                    "Contract_Two year": [1 if contract == "Two year" else 0],
                    "PaperlessBilling_Yes": [1 if paperless_billing == "Yes" else 0],
                }
            )

            try:
                feature_names = preprocessor.get("feature_names", []) if preprocessor else []
                if feature_names:
                    aligned_data = pd.DataFrame(0, index=[0], columns=feature_names)
                    for column, value in customer_data.iloc[0].items():
                        if column in aligned_data.columns:
                            aligned_data.at[0, column] = value
                    customer_scaled = scaler.transform(aligned_data)
                else:
                    customer_scaled = scaler.transform(customer_data)

                prediction = int(model.predict(customer_scaled)[0])
                probability = float(model.predict_proba(customer_scaled)[0][1])
                risk_level, risk_class = risk_state(probability)

                render_section_header("Prediction Result", "Clear output for demos, interviews, and client presentations.")
                render_metrics(
                    [
                        ("Churn probability", f"{probability:.1%}", f"Risk tier: {risk_level}"),
                        ("No-churn probability", f"{1 - probability:.1%}", "Complementary confidence"),
                        ("Prediction", "WILL CHURN" if prediction == 1 else "WILL STAY", "Binary classification result"),
                        ("Risk band", risk_level, "Actionable customer grouping"),
                    ]
                )

                st.write("")
                left_result, right_result = st.columns(2)
                with left_result:
                    render_html(
                        f'''
                        <div class="prediction-card">
                            <div class="section-badge">Risk Assessment</div>
                            <h3 style="margin:0.6rem 0 0;">Status: <span class="{risk_class}">{risk_level}</span></h3>
                            <p style="color:var(--muted);line-height:1.65;margin:0.55rem 0 0;">This customer has a churn probability of {probability:.1%}. The model output is translated into a direct action tier so the outcome is presentation-ready.</p>
                        </div>
                        '''
                    )
                with right_result:
                    render_html(
                        '''
                        <div class="prediction-card">
                            <div class="section-badge">Recommended Actions</div>
                            <p style="color:var(--muted);line-height:1.65;margin:0.55rem 0 0;">The actions below are tailored to the current risk tier and should be used as an operational playbook.</p>
                        </div>
                        '''
                    )
                    if probability >= 0.7:
                        st.error(
                            "High risk detected - this customer is likely to churn. Recommended actions: contact proactively, offer a retention incentive, and follow up within 7 days.",
                            icon="⚠️",
                        )
                    elif probability >= 0.4:
                        st.warning(
                            "Medium risk detected - monitor this customer closely. Recommended actions: add to the watchlist, send a targeted offer, and reinforce service value.",
                            icon="🟡",
                        )
                    else:
                        st.success(
                            "Low risk detected - customer satisfaction appears strong. Recommended actions: maintain service quality, request a survey, and encourage referrals.",
                            icon="✅",
                        )
            except Exception as exc:
                st.error(f"Error making prediction: {exc}")

elif page == "📚 Information":
    render_hero(
        "Project Information",
        "A concise product brief covering the model, the inputs it uses, and the business value delivered by the app.",
        ["Model details", "Feature summary", "Business context"],
        PAGE_HERO_ICONS[page_key],
    )

    render_section_header("Model Overview", "A clear summary of the current production artifact.")
    render_info_cards(
        [
            ("Dataset", "7,043 telecom customers with demographic, service, and billing information.", None),
            ("Best model", f"{metadata.get('model_type', 'AdaBoost')} selected by ROC-AUC after balancing the training set with SMOTE.", None),
            ("Business value", "Helps reduce acquisition spend by identifying high-risk customers before they leave.", None),
            ("Deployment posture", "Optimized for local demo, portfolio showcase, and interviewer walkthroughs.", None),
        ]
    )

    render_section_header("Features Used", "The main customer signals used by the saved model.")
    render_feature_cards(
        [
            ("Contract type", "Month-to-month customers churn significantly more often than long-term subscribers."),
            ("Tenure", "Early lifecycle customers need the most retention attention."),
            ("Monthly charges", "Higher billing correlates with churn risk and price sensitivity."),
            ("Support services", "Tech support and online security reduce churn exposure."),
        ]
    )

    st.write("")
    render_section_header("Performance Metrics", "Current evaluation snapshot from the saved metadata.")
    perf_left, perf_right = st.columns([1.15, 0.85])
    with perf_left:
        perf_fig = go.Figure(
            data=[
                go.Bar(
                    x=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                    y=[
                        metrics.get("Accuracy", 0.0),
                        metrics.get("Precision", 0.0),
                        metrics.get("Recall", 0.0),
                        metrics.get("F1 Score", 0.0),
                        metrics.get("ROC-AUC", 0.0),
                    ],
                    marker_color=["#8ea8ff", "#69d39d", "#f4c26a", "#6fe7d8", "#c28dff"],
                )
            ]
        )
        perf_fig.update_yaxes(gridcolor="rgba(184,206,240,0.14)", range=[0, 1], title_text="Score")
        chart_layout(perf_fig, height=410, showlegend=False)
        st.markdown('<div class="chart-card"><div class="chart-title">Model Performance</div><div class="chart-copy">Current evaluation snapshot for the production artifact.</div></div>', unsafe_allow_html=True)
        st.plotly_chart(perf_fig, width="stretch", config={"displayModeBar": False})
    with perf_right:
        # Prefer loading an up-to-date comparison from models/model_comparison.csv
        import os
        comp_path = Path("models/model_comparison.csv")
        if comp_path.exists():
            try:
                comparison_df = pd.read_csv(comp_path)
                # Format ROC-AUC as strings with 4 decimals for display
                if "ROC-AUC" in comparison_df.columns:
                    comparison_df["ROC-AUC"] = comparison_df["ROC-AUC"].map(lambda v: f"{v:.4f}")
                # Add a status column marking the top model
                if not "Status" in comparison_df.columns:
                    top = comparison_df.iloc[0]["Model"] if not comparison_df.empty else None
                    comparison_df["Status"] = comparison_df["Model"].apply(lambda m: "Best" if m == top else "")
            except Exception:
                # Fallback to small static table if CSV fails to load
                comparison_df = pd.DataFrame(
                    [[metadata.get("model_type", "AdaBoost"), f"{metrics.get('ROC-AUC', 0):.4f}", "Best"]],
                    columns=["Model", "ROC-AUC", "Status"],
                )
        else:
            comparison_df = pd.DataFrame(
                [[metadata.get("model_type", "AdaBoost"), f"{metrics.get('ROC-AUC', 0):.4f}", "Best"]],
                columns=["Model", "ROC-AUC", "Status"],
            )
        st.markdown('<div class="chart-card"><div class="chart-title">Model Comparison</div><div class="chart-copy">Competing models ranked by ROC-AUC.</div></div>', unsafe_allow_html=True)
        st.dataframe(comparison_df, width="stretch", hide_index=True)

    st.write("")
    render_section_header("Business Impact", "Why this dashboard matters in a real analytics workflow.")
    render_info_cards(
        [
            ("Retention focus", "High-risk accounts can be prioritized before churn becomes irreversible.", None),
            ("Revenue protection", "The model helps protect recurring revenue by reducing avoidable customer loss.", None),
            ("Operational clarity", "Teams get a shared, visual language for customer risk and action planning.", None),
            ("Portfolio value", "The UI now looks production-grade for client demos, interviews, and showcases.", None),
        ]
    )

    st.write("")
    render_section_header("Architecture Diagram", "A simple, production-style workflow view for the deployed app.")
    render_html(f'''
        <div class="architecture-grid">
            <div class="architecture-node">
                <div class="node-kicker">Step 1</div>
                <div class="node-title">Customer Data</div>
                <p class="node-copy">Telecom profile, billing, and service usage inputs enter the application.</p>
            </div>
            <div class="architecture-node">
                <div class="node-kicker">Step 2</div>
                <div class="node-title">Preprocessing</div>
                <p class="node-copy">Inputs are aligned with the trained feature schema and scaled consistently.</p>
            </div>
            <div class="architecture-node">
                <div class="node-kicker">Step 3</div>
                <div class="node-title">{metadata.get('model_type', 'AdaBoost')}</div>
                <p class="node-copy">The saved production model scores churn probability using the trained artifact.</p>
            </div>
            <div class="architecture-node">
                <div class="node-kicker">Step 4</div>
                <div class="node-title">Retention Action</div>
                <p class="node-copy">Results are translated into risk levels, recommendations, and dashboard insights.</p>
            </div>
        </div>
        '''
    )

st.divider()
st.markdown('<div class="footer-note">Built for customer retention analytics • Streamlit + scikit-learn</div>', unsafe_allow_html=True)
