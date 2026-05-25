"""
Streamlit App - Customer Churn Prediction

Interactive web application for predicting customer churn.

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from html import escape
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_manager import ModelManager
from src.models.predict import ModelPredictor
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="◼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111d;
            --bg-alt: #0b1a2f;
            --sidebar-bg: #04070c;
            --sidebar-surface: rgba(10, 14, 20, 0.96);
            --main-surface: rgba(12, 23, 41, 0.70);
            --main-surface-strong: rgba(19, 38, 67, 0.84);
            --panel: rgba(18, 30, 52, 0.76);
            --panel-strong: rgba(22, 39, 67, 0.92);
            --panel-soft: rgba(34, 56, 92, 0.38);
            --border: rgba(173, 194, 230, 0.16);
            --border-strong: rgba(173, 194, 230, 0.28);
            --text: #edf4ff;
            --muted: #adc0dd;
            --accent: #67f0de;
            --accent-2: #88a8ff;
            --accent-3: #c78cff;
            --danger: #ff7d7d;
            --success: #67d29f;
            --warning: #f3bd62;
            --shadow: 0 28px 76px rgba(0, 0, 0, 0.40);
            --shadow-soft: 0 18px 44px rgba(0, 0, 0, 0.24);
            --sidebar-shadow: 16px 0 38px rgba(0, 0, 0, 0.34);
            --radius-xl: 30px;
            --radius-lg: 22px;
            --radius-md: 16px;
            --radius-sm: 12px;
        }

        .stApp {
            background:
                linear-gradient(90deg, rgba(4, 7, 12, 0.98) 0%, rgba(4, 7, 12, 0.98) 24%, rgba(6, 15, 28, 0.92) 24%, rgba(10, 30, 53, 0.96) 100%),
                radial-gradient(circle at 72% 10%, rgba(104, 235, 226, 0.16), transparent 22%),
                radial-gradient(circle at 85% 34%, rgba(136, 168, 255, 0.16), transparent 26%),
                radial-gradient(circle at 52% 60%, rgba(199, 140, 255, 0.08), transparent 40%);
            color: var(--text);
            font-family: "Aptos", "Segoe UI Variable", "Segoe UI", sans-serif;
        }

        section[data-testid="stMain"] {
            background:
                radial-gradient(circle at top right, rgba(104, 235, 226, 0.10), transparent 24%),
                radial-gradient(circle at 60% 0%, rgba(136, 168, 255, 0.14), transparent 28%),
                linear-gradient(180deg, rgba(12, 25, 44, 0.30), rgba(7, 16, 29, 0.06));
            flex: 1 1 0% !important;
            min-width: 0 !important;
            transition: max-width 220ms ease, padding 220ms ease, margin 220ms ease;
        }

        section[data-testid="stMain"] > div {
            background: transparent;
        }

        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2.2rem;
            max-width: 1500px;
        }

        .block-container > div:first-child {
            padding-top: 0.2rem;
        }

        h1, h2, h3, h4, p, label, div {
            color: var(--text);
        }

        .stApp header {
            background: linear-gradient(90deg, rgba(10,14,20,0.36), rgba(6,12,20,0.18));
            border-bottom: 1px solid rgba(255,255,255,0.03);
            backdrop-filter: blur(6px);
            padding: 0.35rem 0.6rem;
        }

        .stApp [data-testid="stToolbar"] {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
            padding: 0 0.6rem;
            opacity: 1;
            pointer-events: auto;
        }

        .stApp [data-testid="collapsedControl"] {
            background: linear-gradient(135deg, rgba(136, 168, 255, 0.24), rgba(104, 235, 226, 0.14));
            border: 1px solid rgba(173, 194, 230, 0.22);
            border-radius: 12px;
            box-shadow: var(--shadow-soft);
            width: 2.1rem;
            height: 2.1rem;
            color: var(--text);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, filter 180ms ease;
        }

        .stApp [data-testid="collapsedControl"]:hover {
            transform: translateY(-1px);
            border-color: rgba(104, 235, 226, 0.34);
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.26);
            filter: brightness(1.05);
        }

        .stApp [data-testid="collapsedControl"] svg {
            width: 1rem;
            height: 1rem;
            fill: currentColor;
        }

        /* Make the floating collapse control more visible */
        .stApp [data-testid="collapsedControl"] {
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(95,125,255,0.14), rgba(67,240,222,0.12));
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 10px 28px rgba(6, 12, 24, 0.56);
            color: var(--text);
            z-index: 1400;
        }

        .stApp [data-testid="collapsedControl"] svg {
            width: 1.15rem;
            height: 1.15rem;
            color: var(--text);
            opacity: 0.98;
        }

        /* Position collapse control at the sidebar edge for visibility */
        .stApp [data-testid="collapsedControl"] {
            position: absolute;
            top: 10px;
            right: -34px;
            transform: none;
            transition: transform 160ms ease, right 160ms ease;
            filter: drop-shadow(0 16px 44px rgba(2,8,20,0.82));
            backdrop-filter: blur(10px);
            z-index: 1800;
            border: 2px solid rgba(255,255,255,0.10);
            background: radial-gradient(circle at 40% 35%, rgba(255,255,255,0.06), rgba(255,255,255,0.02)), linear-gradient(135deg, rgba(95,125,255,0.28), rgba(67,240,222,0.20));
            padding: 6px;
        }

        /* When sidebar is collapsed, keep control visible inside compact rail */
        section[data-testid="stSidebar"][aria-expanded="false"] .stButton, /* fallback */
        section[data-testid="stSidebar"][aria-expanded="false"] .css-1v3fvcr { }

        /* Ensure icon paths are white and high contrast */
        .stApp [data-testid="collapsedControl"] svg path,
        .stApp button[data-testid="stExpandSidebarButton"] svg path {
            fill: #ffffff !important;
            stroke: rgba(0,0,0,0.26) !important;
            stroke-width: 0.6px !important;
            vector-effect: non-scaling-stroke !important;
            opacity: 1 !important;
        }

        /* Strong hover/focus states so the arrow is always visible */
        .stApp [data-testid="collapsedControl"]:hover,
        .stApp button[data-testid="stExpandSidebarButton"]:hover {
            transform: translateY(-2px) scale(1.04);
            right: -18px;
            box-shadow: 0 18px 40px rgba(6,12,24,0.56);
        }

        .stApp button[data-testid="stExpandSidebarButton"],
        .stApp button[data-testid="stBaseButton-headerNoPadding"] {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.15rem !important;
            height: 2.15rem !important;
            min-width: 2.15rem !important;
            min-height: 2.15rem !important;
            padding: 0 !important;
            border-radius: 999px !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            background: linear-gradient(135deg, rgba(95, 125, 255, 0.20), rgba(67, 240, 222, 0.14)) !important;
            box-shadow: 0 10px 26px rgba(6, 12, 24, 0.46) !important;
            color: #f7fbff !important;
            overflow: hidden;
        }

        .stApp button[data-testid="stExpandSidebarButton"]:hover,
        .stApp button[data-testid="stBaseButton-headerNoPadding"]:hover {
            transform: translateY(-1px) scale(1.02);
            filter: brightness(1.06);
            border-color: rgba(95,125,255,0.36) !important;
            box-shadow: 0 14px 30px rgba(6, 12, 24, 0.56) !important;
        }

        .stApp button[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
        .stApp button[data-testid="stBaseButton-headerNoPadding"] [data-testid="stIconMaterial"] {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #f8fdff !important;
            opacity: 1 !important;
            font-size: 1.2rem !important;
            font-weight: 800 !important;
            line-height: 1 !important;
            letter-spacing: 0;
            text-shadow: 0 2px 6px rgba(0, 0, 0, 0.32);
        }

        .stApp button[data-testid="stExpandSidebarButton"] span,
        .stApp button[data-testid="stBaseButton-headerNoPadding"] span {
            color: #f8fdff !important;
            opacity: 1 !important;
        }

        .stSidebar {
            /* Dark professional sidebar to match main theme */
            background: linear-gradient(180deg, var(--sidebar-surface), rgba(8,12,20,0.72));
            border-right: 1px solid rgba(255, 255, 255, 0.03);
            box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.02), var(--sidebar-shadow);
            color: var(--text);
            backdrop-filter: blur(12px);
        }

        section[data-testid="stSidebar"] {
            width: 19rem;
            min-width: 19rem;
            max-width: 19rem;
            flex: 0 0 19rem !important;
            background: linear-gradient(180deg, var(--sidebar-bg), var(--sidebar-surface));
            transition: width 220ms ease, min-width 220ms ease, max-width 220ms ease;
            position: relative;
        }

        /* Compact collapsed sidebar: icon-only rail with hover peek */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 4.8rem !important;
            min-width: 4.8rem !important;
            max-width: 4.8rem !important;
            flex: 0 0 4.8rem !important;
            display: block !important;
            overflow: visible !important;
            transition: width 220ms ease, min-width 220ms ease, max-width 220ms ease;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] > div {
            width: 4.8rem !important;
            padding: 0.45rem 0.25rem !important;
            overflow: visible !important;
        }

        /* Center icons and hide textual labels when collapsed */
        section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stRadio"] label {
            justify-content: center;
            padding: 0.6rem !important;
            border-radius: 10px;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stRadio"] label p,
        section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stRadio"] label span {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Keep icons visible and centered */
        section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stRadio"] label svg {
            margin: 0 !important;
            width: 20px !important;
            height: 20px !important;
        }

        /* Hover peek: expand to full sidebar on mouseover */
        section[data-testid="stSidebar"][aria-expanded="false"]:hover {
            width: 17rem !important;
            min-width: 17rem !important;
        }

        /* Allow main content to respect the compact rail width */
        section[data-testid="stSidebar"][aria-expanded="false"] ~ section[data-testid="stMain"] {
            margin-left: 4.8rem !important;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] ~ section[data-testid="stMain"] .block-container {
            max-width: calc(100% - 4.8rem) !important;
            margin-left: auto;
            margin-right: auto;
        }

        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 19rem !important;
            min-width: 19rem !important;
            max-width: 19rem !important;
            flex: 0 0 19rem !important;
        }

        section[data-testid="stSidebar"] > div {
            padding: 1.15rem 1.05rem 1.2rem;
        }

        .stSidebar [data-testid="stMarkdownContainer"] p,
        .stSidebar label,
        .stSidebar .stCaption,
        .stSidebar .stRadio {
            color: var(--text);
            opacity: 0.95 !important;
        }

        .stSidebar [data-testid="stCaptionContainer"] {
            opacity: 0.92 !important;
            color: var(--muted) !important;
        }

        .stSidebar [data-testid="stCaptionContainer"] p {
            color: var(--muted) !important;
            opacity: 0.92 !important;
            font-weight: 600;
        }

        .stSidebar [data-testid="stMarkdownContainer"] p {
            color: var(--muted) !important;
            opacity: 0.92 !important;
            font-weight: 600;
        }

        .stSidebar [data-testid="stMarkdownContainer"] h1,
        .stSidebar [data-testid="stMarkdownContainer"] h2,
        .stSidebar [data-testid="stMarkdownContainer"] h3,
        .stSidebar [data-testid="stMarkdownContainer"] h4,
        .stSidebar [data-testid="stMarkdownContainer"] strong,
        .stSidebar [data-testid="stMarkdownContainer"] span {
            color: var(--text) !important;
            opacity: 0.96 !important;
        }

        .stSidebar [data-testid="stRadio"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 12px;
            padding: 0.6rem;
            box-shadow: 0 8px 20px rgba(2, 8, 20, 0.28);
        }

        .stSidebar [data-testid="stRadio"] div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }

        .stSidebar [data-testid="stRadio"] label {
            display: flex;
            align-items: center;
            gap: 0.78rem;
            min-height: 2.85rem;
            padding: 0.62rem 0.7rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.03);
            transition: transform 180ms ease, background 180ms ease, border-color 180ms ease, box-shadow 180ms ease, filter 180ms ease;
            font-size: 0.98rem;
            font-weight: 700;
            background: transparent;
            color: var(--text) !important;
        }

        .stSidebar [data-testid="stRadio"] label:hover {
            background: rgba(104, 235, 226, 0.06);
            border-color: rgba(104, 235, 226, 0.10);
            transform: translateX(2px) translateY(-1px);
            filter: brightness(1.02);
        }

        .stSidebar [data-testid="stRadio"] label[data-checked="true"] {
            background: linear-gradient(135deg, rgba(95, 125, 255, 0.10), rgba(67, 240, 222, 0.06));
            border-color: rgba(95, 125, 255, 0.28);
            box-shadow: 0 10px 26px rgba(12, 22, 36, 0.28);
        }

        .stSidebar [data-testid="stRadio"] label p {
            margin: 0;
            line-height: 1.15;
            color: var(--text) !important;
            font-weight: 700;
            white-space: normal;
            opacity: 0.98 !important;
        }

        .stSidebar [data-testid="stRadio"] label span {
            color: var(--text) !important;
            font-weight: 700;
            opacity: 0.98 !important;
        }

        .stSidebar [data-testid="stRadio"] label svg {
            width: 18px;
            height: 18px;
            min-width: 18px;
            min-height: 18px;
            color: var(--accent-2);
            fill: currentColor;
            opacity: 0.98;
        }

        .stSidebar [data-testid="stRadio"] svg {
            width: 18px;
            height: 18px;
            margin-top: 1px;
        }

        .stSidebar hr {
            border-color: rgba(255,255,255,0.04);
        }

        .app-shell {
            border: 1px solid rgba(173, 194, 230, 0.14);
            background:
                radial-gradient(circle at top right, rgba(104, 235, 226, 0.10), transparent 22%),
                linear-gradient(180deg, rgba(14, 29, 50, 0.70), rgba(9, 18, 32, 0.90));
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow);
            padding: 1.15rem 1.35rem;
            margin-bottom: 1.25rem;
            backdrop-filter: blur(18px);
        }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(173, 194, 230, 0.16);
            border-radius: 30px;
            background:
                radial-gradient(circle at top right, rgba(104, 235, 226, 0.20), transparent 26%),
                radial-gradient(circle at left center, rgba(136, 168, 255, 0.20), transparent 30%),
                linear-gradient(135deg, rgba(14, 28, 50, 0.96), rgba(8, 16, 30, 0.90));
            box-shadow: var(--shadow);
            padding: 1.55rem 1.65rem 1.35rem;
            margin: 0 0 1rem 0;
            backdrop-filter: blur(18px);
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.6fr 0.95fr;
            gap: 1rem;
            align-items: stretch;
        }

        .hero-panel {
            border-radius: 26px;
            border: 1px solid rgba(173, 194, 230, 0.14);
            background: linear-gradient(135deg, rgba(15, 31, 55, 0.92), rgba(9, 16, 29, 0.88));
            box-shadow: var(--shadow-soft);
            padding: 1.15rem 1.2rem;
            backdrop-filter: blur(16px);
            min-height: 100%;
        }

        .hero-panel h2 {
            margin: 0 0 0.45rem;
            font-size: clamp(1.8rem, 2.5vw, 3rem);
            line-height: 1.04;
            letter-spacing: -0.03em;
        }

        .hero-panel .hero-copy {
            color: #c4d6f0;
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 1rem;
        }

        .hero-panel .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            border: 1px solid rgba(127, 156, 255, 0.18);
            background: rgba(127, 156, 255, 0.12);
            color: #dce5ff;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero-panel .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.9rem;
        }

        .hero-panel .hero-tag {
            border-radius: 999px;
            padding: 0.38rem 0.75rem;
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.15);
            color: #dde7fb;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .hero-side-panel {
            border-radius: 26px;
            border: 1px solid rgba(173, 194, 230, 0.14);
            background: radial-gradient(circle at top right, rgba(104, 235, 226, 0.14), transparent 35%), linear-gradient(180deg, rgba(17, 31, 53, 0.88), rgba(9, 15, 28, 0.84));
            box-shadow: var(--shadow-soft);
            padding: 1rem 1.05rem;
            backdrop-filter: blur(16px);
        }

        .hero-side-panel h3 {
            margin: 0 0 0.65rem;
            font-size: 1rem;
            letter-spacing: 0.02em;
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: auto -60px -120px auto;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(87, 227, 207, 0.22), transparent 68%);
            pointer-events: none;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            background: rgba(127, 156, 255, 0.11);
            border: 1px solid rgba(127, 156, 255, 0.20);
            color: #d8e1ff;
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
        }

        .hero-title {
            margin-top: 0.9rem;
            margin-bottom: 0.45rem;
            font-size: clamp(2.1rem, 3.1vw, 3.5rem);
            font-weight: 800;
            line-height: 1.02;
            letter-spacing: -0.03em;
        }

        .hero-subtitle {
            max-width: 760px;
            color: #c4d6f0;
            font-size: 1.02rem;
            line-height: 1.72;
            margin-bottom: 0.95rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 0.95rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.8rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.16);
            color: #dfe8fb;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 800;
            margin: 0.3rem 0 0.3rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.97rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        .subsection-note {
            color: var(--muted);
            font-size: 0.95rem;
            gap: 0.42rem;
            margin: 0.2rem 0 1.15rem;
        }

        .card {
            background: linear-gradient(180deg, rgba(16, 31, 53, 0.88), rgba(10, 18, 32, 0.84));
            font-size: 1.6rem;
            border-radius: 24px;
            padding: 1.05rem 1.1rem;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(16px);
        }

        .metric-grid {
            line-height: 1.65;
            max-width: 74ch;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));

        .section-badge + .section-header {
            margin-top: 0;
        }

        .hero-shell {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.85fr);
            gap: 1rem;
            align-items: stretch;
            padding: 1.35rem 1.4rem;
            border-radius: 30px;
            border: 1px solid rgba(173, 194, 230, 0.16);
            background:
                radial-gradient(circle at top right, rgba(104, 235, 226, 0.14), transparent 26%),
                radial-gradient(circle at bottom left, rgba(136, 168, 255, 0.10), transparent 28%),
                linear-gradient(180deg, rgba(16, 30, 52, 0.90), rgba(10, 18, 32, 0.86));
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .hero-copy {
            min-width: 0;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 0.7rem;
            padding: 0.34rem 0.78rem;
            border-radius: 999px;
            border: 1px solid rgba(104, 235, 226, 0.22);
            background: rgba(104, 235, 226, 0.10);
            color: #d9f7f2;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .hero-copy h1 {
            margin: 0;
            font-size: clamp(2rem, 3.1vw, 3.3rem);
            line-height: 1.04;
            letter-spacing: -0.03em;
        }

        .hero-copy p {
            margin: 0.75rem 0 0;
            max-width: 68ch;
            color: var(--muted);
            line-height: 1.72;
            font-size: 1rem;
        }

        .hero-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .hero-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.48rem 0.78rem;
            border-radius: 999px;
            border: 1px solid rgba(173, 194, 230, 0.16);
            background: rgba(255, 255, 255, 0.04);
            color: #e6eefc;
            font-size: 0.83rem;
            font-weight: 700;
            line-height: 1;
        }

        .hero-aside {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.9rem;
            min-width: 0;
            padding: 1rem 1rem 1.05rem;
            border-radius: 24px;
            border: 1px solid rgba(173, 194, 230, 0.14);
            background:
                radial-gradient(circle at top right, rgba(136, 168, 255, 0.14), transparent 30%),
                linear-gradient(180deg, rgba(18, 34, 58, 0.92), rgba(12, 20, 35, 0.84));
            box-shadow: var(--shadow-soft);
        }

        .hero-aside__label {
            color: #bfd0ea;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-aside__value {
            font-size: 1.45rem;
            font-weight: 800;
            line-height: 1.12;
            color: #f8fbff;
            letter-spacing: -0.02em;
        }

        .hero-aside__note {
            color: var(--muted);
            line-height: 1.6;
            margin: 0;
        }

        .card-grid {
            display: grid;
            gap: 1rem;
        }

        .card-grid--metrics {
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        }

        .card-grid--features {
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        }

        .card-grid--stack {
            grid-template-columns: 1fr;
            gap: 0.78rem;
        }

        .streamlit-card-shell {
            position: relative;
            border: 1px solid rgba(173, 194, 230, 0.16);
            border-radius: 22px;
            background:
                radial-gradient(circle at top right, rgba(104, 235, 226, 0.12), transparent 28%),
                linear-gradient(180deg, rgba(17, 34, 60, 0.90), rgba(10, 18, 32, 0.84));
            padding: 1rem 1.05rem;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(16px);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, background 180ms ease;
            height: 100%;
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 0.35rem;
            overflow: hidden;
        }

        .streamlit-card-shell.metric-card {
            min-height: 126px;
        }

        .streamlit-card-shell.feature-card {
            min-height: 154px;
        }

        .streamlit-card-shell.compact {
            min-height: 106px;
            padding: 0.88rem 0.95rem;
        }
            gap: 1rem;
        }

        .accent-card {
            border-radius: 22px;
            padding: 1.1rem 1.15rem;
            border: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(104, 235, 226, 0.12), rgba(136, 168, 255, 0.08));
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
        }

        .feature-card {
            background: linear-gradient(180deg, rgba(16, 31, 53, 0.88), rgba(10, 18, 32, 0.82));
            border: 1px solid rgba(173, 194, 230, 0.14);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            min-height: 152px;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(16px);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }

        .feature-card:hover {
            transform: translateY(-3px);
            border-color: rgba(104, 235, 226, 0.28);
            box-shadow: 0 20px 48px rgba(0, 0, 0, 0.30);
        }

        .feature-card h4 {
            margin: 0 0 0.35rem;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.25;
        }

        .feature-card p {
            margin: 0;
            color: var(--muted);
            line-height: 1.6;
        }

        .section-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.34rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(127, 156, 255, 0.20);
            background: rgba(127, 156, 255, 0.11);
            color: #c9d4ff;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .section-header {
            display: flex;
            flex-direction: column;
            gap: 0.28rem;
            margin: 0.2rem 0 1rem;
        }

        .section-header h2 {
            margin: 0;
            font-size: 1.52rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .section-header p {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
        }

        .chart-panel {
            border-radius: 22px;
            border: 1px solid rgba(173, 194, 230, 0.14);
            background: linear-gradient(180deg, rgba(16, 31, 53, 0.88), rgba(10, 18, 32, 0.82));
            padding: 1rem 1rem 0.4rem;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(16px);
        }

        .chart-panel h4 {
            margin: 0 0 0.5rem;
            font-size: 1rem;
            font-weight: 800;
        }

        .prediction-panel {
            border-radius: 24px;
            border: 1px solid rgba(173, 194, 230, 0.14);
            background: linear-gradient(180deg, rgba(16, 31, 53, 0.88), rgba(10, 18, 32, 0.82));
            padding: 1.1rem;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(16px);
        }

        .prediction-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.9rem;
        }

        .input-group {
            margin-bottom: 0.95rem;
        }

        .input-group h4 {
            margin: 0 0 0.65rem;
            font-size: 1rem;
            font-weight: 800;
        }

        .input-group .stSelectbox,
        .input-group .stNumberInput {
            margin-bottom: 0.5rem;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] {
            background: rgba(255, 255, 255, 0.04) !important;
            border-color: rgba(173, 194, 230, 0.16) !important;
            border-radius: 14px !important;
        }

        .stNumberInput input,
        .stTextInput input {
            color: var(--text) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(173, 194, 230, 0.14);
            border-radius: 18px;
            padding: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border-radius: 14px !important;
            color: var(--muted) !important;
            font-weight: 700 !important;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(136, 168, 255, 0.22), rgba(104, 235, 226, 0.12)) !important;
            color: var(--text) !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #f3f7ff !important;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            border-color: rgba(153, 175, 208, 0.14) !important;
        }

        .stDataFrame, .stTable {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(173, 194, 230, 0.14);
        }

        .input-card .stSelectbox, .input-card .stNumberInput {
            margin-bottom: 0.4rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(136, 168, 255, 0.24);
            background: linear-gradient(135deg, rgba(136, 168, 255, 0.22), rgba(104, 235, 226, 0.16));
            color: white;
            font-weight: 700;
            padding: 0.7rem 1rem;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, filter 180ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(104, 235, 226, 0.42);
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.26);
            filter: brightness(1.05);
        }

        .stInfo, .stSuccess, .stWarning, .stError {
            border-radius: 18px;
            border: 1px solid rgba(173, 194, 230, 0.14);
        }

        .stAlert {
            border-radius: 18px;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.18);
        }

        .stInfo {
            background: rgba(121, 154, 255, 0.10) !important;
        }

        .stSuccess {
            background: rgba(103, 210, 159, 0.10) !important;
        }

        .stWarning {
            background: rgba(243, 189, 98, 0.10) !important;
        }

        .stError {
            background: rgba(255, 125, 125, 0.10) !important;
        }

        .subtle-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(173, 194, 230, 0.32), transparent);
            margin: 1.1rem 0;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
        }

        .info-tile {
            background: linear-gradient(180deg, rgba(16, 31, 53, 0.88), rgba(10, 18, 32, 0.82));
            border: 1px solid rgba(173, 194, 230, 0.14);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(16px);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }

        .info-tile:hover {
            transform: translateY(-3px);
            border-color: rgba(87, 227, 207, 0.24);
            box-shadow: 0 20px 48px rgba(0, 0, 0, 0.30);
        }

        .info-tile h4 {
            margin: 0 0 0.35rem;
            font-size: 1rem;
            font-weight: 800;
        }

        .info-tile p {
            margin: 0;
            color: var(--muted);
            line-height: 1.6;
        }

        .footer-note {
            text-align: center;
            color: var(--muted);
            padding: 0.7rem 0 0.1rem;
            font-size: 0.92rem;
        }

        .streamlit-card-shell:hover {
            transform: translateY(-3px);
            border-color: rgba(104, 235, 226, 0.30);
            box-shadow: 0 20px 48px rgba(0, 0, 0, 0.30);
        }

        .streamlit-card-shell.metric-card:hover,
        .streamlit-card-shell.feature-card:hover,
        .streamlit-card-shell.compact:hover {
            background:
                radial-gradient(circle at top right, rgba(104, 235, 226, 0.16), transparent 28%),
                linear-gradient(180deg, rgba(20, 38, 66, 0.96), rgba(12, 22, 38, 0.92));
        }

        .streamlit-card-shell h4,
        .streamlit-card-shell .shell-label {
            margin: 0;
            color: #dbe8fb !important;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            line-height: 1.2;
            word-break: normal;
            overflow-wrap: normal;
        }

        .streamlit-card-shell .shell-value {
            color: #f8fbff !important;
            font-weight: 800;
            font-size: clamp(1.08rem, 1.9vw, 1.95rem);
            line-height: 1.08;
            letter-spacing: -0.02em;
            white-space: normal;
            word-break: normal;
            overflow-wrap: normal;
        }

        .streamlit-card-shell .shell-note {
            color: #c5d6ea !important;
            font-size: 0.88rem;
            line-height: 1.5;
            margin-top: auto;
        }

        .streamlit-card-shell.metric-card .shell-value {
            font-size: clamp(1.5rem, 2vw, 2.35rem);
        }

        .streamlit-card-shell.feature-card .shell-value {
            font-size: 1.02rem;
            line-height: 1.52;
        }

        .streamlit-card-shell.feature-card .shell-note {
            margin-top: 0;
        }

        .streamlit-card-shell.compact .shell-label {
            font-size: 0.72rem;
        }

        .streamlit-card-shell.compact .shell-value {
            font-size: clamp(1.02rem, 1.45vw, 1.34rem);
            line-height: 1.1;
        }

        .streamlit-card-shell.compact .shell-note {
            font-size: 0.82rem;
        }

        .stSidebar .card-grid--stack {
            gap: 0.72rem;
        }

        .kpi-accent {
            position: relative;
        }

        .kpi-accent::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 22px;
            padding: 1px;
            background: linear-gradient(135deg, rgba(127, 156, 255, 0.55), rgba(87, 227, 207, 0.30), rgba(199, 140, 255, 0.35));
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
        }

        .stMetric {
            background: linear-gradient(180deg, rgba(16, 31, 53, 0.90), rgba(10, 18, 32, 0.84)) !important;
            border: 1px solid rgba(173, 194, 230, 0.14) !important;
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem !important;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(16px);
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
            height: 100%;
        }

        .stMetric:hover {
            transform: translateY(-3px);
            border-color: rgba(104, 235, 226, 0.24) !important;
            box-shadow: 0 20px 48px rgba(0, 0, 0, 0.30);
        }

        .stMetric [data-testid="metric-container"] {
            width: 100%;
        }

        .stMetric label,
        .stMetric [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            white-space: normal !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
        }

        .stMetric [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-weight: 800 !important;
            font-size: clamp(1.35rem, 2.3vw, 2.35rem) !important;
            line-height: 1.08 !important;
            white-space: normal !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
            min-height: 2.6rem;
        }

        .stMetric [data-testid="metric-container"] {
            width: 100%;
            overflow: visible !important;
        }

        .stMetric [data-testid="stMetricValue"],
        .stMetric [data-testid="stMetricLabel"] {
            display: block;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        .stSidebar .stMetric {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(244, 248, 255, 0.62)) !important;
            border: 1px solid rgba(136, 152, 180, 0.14) !important;
            box-shadow: 0 12px 30px rgba(12, 24, 44, 0.08);
            padding: 0.78rem 0.82rem 0.74rem !important;
        }

        .stSidebar .stMetric [data-testid="stMetricLabel"] {
            font-size: 0.74rem !important;
            color: #58708e !important;
        }

        .stSidebar .stMetric [data-testid="stMetricValue"] {
            font-size: clamp(0.92rem, 1.55vw, 1.15rem) !important;
            line-height: 1.14 !important;
            color: #102133 !important;
        }

        .stSidebar .stMetric [data-testid="stMetricValue"] > div {
            white-space: normal !important;
        }

        .stMetric [data-testid="stMetricValue"] > div {
            white-space: normal !important;
        }

        .stMetric [data-testid="stMetricDelta"] {
            color: #c6d4f0 !important;
            white-space: normal !important;
        }

        @media (max-width: 1100px) {
            .metric-grid, .info-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .feature-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .hero-grid {
                grid-template-columns: 1fr;
            }

            .hero-shell {
                grid-template-columns: 1fr;
            }

            .card-grid--metrics,
            .card-grid--features {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 760px) {
            .block-container {
                padding: 1.05rem 1.1rem;
                padding-right: 1rem;
            }
                min-height: 138px;
                padding: 1.1rem 1.15rem;
                padding: 1.1rem 1rem;
                border-radius: 24px;
            }

                font-size: 0.86rem;
                grid-template-columns: 1fr;
            }

            .hero-panel,
                font-size: clamp(1.18rem, 2.1vw, 2.15rem);
                padding: 0.95rem;
                border-radius: 22px;
            }
                font-size: 0.92rem;
            .metric-grid, .info-grid {
                grid-template-columns: 1fr;
            }

            .feature-grid {
                font-size: clamp(1.68rem, 2.25vw, 2.55rem);
            }

                font-size: 0.76rem;
            .card-grid--features {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 1.8rem;
            }

            .section-header h2 {
                font-size: 1.35rem;
            }

            .hero-shell {
                padding: 1rem;
                border-radius: 24px;
            }

            .hero-aside {
                padding: 0.9rem;
            }

            .chart-panel,
            .prediction-panel {
                padding: 0.9rem;
            }

            .stSidebar [data-testid="stRadio"] label {
                min-height: 2.8rem;
                padding: 0.66rem 0.78rem;
            }

            .stSidebar [data-testid="stRadio"] div[role="radiogroup"] {
                gap: 0.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, note: str | None = None, compact: bool = False, kind: str = "metric") -> None:
    note_html = f'<div class="shell-note">{escape(str(note))}</div>' if note else ""
    classes = ["streamlit-card-shell", "kpi-accent", f"{kind}-card"]
    if compact:
        classes = ["streamlit-card-shell", "kpi-accent", "compact"]
    st.markdown(
        f"""
        <div class="{' '.join(classes)}">
            <div class="shell-label">{escape(str(label))}</div>
            <div class="shell-value">{escape(str(value))}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, icon: str, chips: list[str] | None = None) -> None:
    st.markdown(f"# {icon} {title}")
    st.caption(subtitle)
    if chips:
        st.markdown(
            "<div class='hero-chips'>" + "".join(f"<span class='hero-chip'>{escape(str(chip))}</span>" for chip in chips) + "</div>",
            unsafe_allow_html=True,
        )


def render_metric_grid(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items):
        with column:
            render_stat_card(label, value, note)


def render_metric_strip(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items):
        with column:
            render_stat_card(label, value, note)


def render_snapshot_stack(items: list[tuple[str, str]]) -> None:
    for label, value in items:
        render_stat_card(label, value, compact=True)


def render_info_tiles(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (title, body) in zip(columns, items):
        with column:
            render_stat_card(title, body, compact=False, kind="feature")


def render_section_header(title: str, subtitle: str | None = None, badge: str | None = None) -> None:
    if badge:
        st.caption(badge)
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1560px;
        }

        .hero-copy h1 {
            margin: 0;
            font-size: clamp(2rem, 3vw, 3.4rem);
            line-height: 1.04;
            letter-spacing: -0.03em;
        }

        .hero-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .hero-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.45rem 0.78rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.09);
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: #dce7fb;
            font-size: 0.8rem;
            font-weight: 700;
        }

        .streamlit-card-shell {
            background:
                radial-gradient(circle at top right, rgba(104, 235, 226, 0.16), transparent 30%),
                linear-gradient(180deg, rgba(18, 34, 58, 0.96), rgba(10, 18, 32, 0.92));
            border: 1px solid rgba(173, 194, 230, 0.16);
            border-radius: 22px;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(18px);
        }

        .streamlit-card-shell.metric-card {
            min-height: 140px;
            padding: 1.1rem 1.15rem;
        }

        .streamlit-card-shell.feature-card {
            min-height: 154px;
            padding: 1rem 1.05rem;
        }

        .streamlit-card-shell.compact {
            min-height: 106px;
            padding: 0.88rem 0.95rem;
        }

        .streamlit-card-shell .shell-label {
            color: #dbe8fb !important;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            line-height: 1.2;
        }

        .streamlit-card-shell .shell-value {
            color: #f8fbff !important;
            font-weight: 800;
            font-size: clamp(1.15rem, 2.1vw, 2.2rem);
            line-height: 1.08;
            letter-spacing: -0.02em;
            word-break: normal;
            overflow-wrap: normal;
        }

        .streamlit-card-shell.metric-card .shell-value {
            font-size: clamp(1.7rem, 2.3vw, 2.6rem);
        }

        .streamlit-card-shell.feature-card .shell-value {
            font-size: 1rem;
            line-height: 1.48;
        }

        .streamlit-card-shell .shell-note {
            color: #c5d6ea !important;
            font-size: 0.9rem;
            line-height: 1.48;
            margin-top: auto;
        }

        .streamlit-card-shell:hover {
            transform: translateY(-3px);
            border-color: rgba(104, 235, 226, 0.30);
            box-shadow: 0 20px 48px rgba(0, 0, 0, 0.30);
        }

        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1rem;
        }

        .model-grid {
            display: grid;
            grid-template-columns: 1.25fr 0.75fr;
            gap: 1rem;
        }

        .timeline-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 0.9rem;
        }

        @media (max-width: 1100px) {
            .hero-shell,
            .model-grid {
                grid-template-columns: 1fr;
            }
        }
        /* Strong override for collapse control to ensure visibility */
        .stApp [data-testid="collapsedControl"] {
            position: absolute !important;
            top: 12px !important;
            right: -10px !important;
            width: 36px !important;
            height: 36px !important;
            padding: 6px !important;
            border-radius: 999px !important;
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)) !important;
            border: 2px solid rgba(95,125,255,0.28) !important;
            box-shadow: 0 22px 48px rgba(2,8,20,0.84) !important;
            z-index: 9999 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .stApp [data-testid="collapsedControl"] svg,
        .stApp [data-testid="collapsedControl"] svg path {
            width: 18px !important;
            height: 18px !important;
            fill: #ffffff !important;
            stroke: rgba(0,0,0,0.28) !important;
            stroke-width: 0.5px !important;
        }

        section[data-testid="stSidebar"] { position: relative !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig: go.Figure, *, height: int = 360, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        showlegend=showlegend,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=12, r=12, t=16, b=12),
        font=dict(color='#e6eefc', family='Aptos, Segoe UI Variable, Segoe UI, sans-serif'),
        legend=dict(orientation='h', yanchor='bottom', y=-0.18, x=0.02),
    )
    return fig


inject_styles()
inject_dashboard_styles()
def inject_sidebar_enhancements() -> None:
    st.markdown(
        """
        <style>
        /* Tooltip for collapsed sidebar labels */
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            position: relative;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-tooltip]::after {
            content: attr(data-tooltip);
            position: absolute;
            left: 100%;
            top: 50%;
            transform: translate(12px, -50%) scale(0.98);
            background: rgba(8, 12, 20, 0.96);
            color: var(--text);
            padding: 0.45rem 0.7rem;
            border-radius: 8px;
            box-shadow: 0 10px 26px rgba(2,8,20,0.48);
            white-space: nowrap;
            font-size: 0.86rem;
            font-weight: 700;
            opacity: 0;
            pointer-events: none;
            transition: opacity 140ms ease, transform 140ms ease;
            z-index: 1200;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stRadio"] label:hover::after {
            opacity: 1;
            transform: translate(16px, -50%) scale(1);
        }

        /* Focus ring for keyboard accessibility */
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:focus-visible,
        section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
            outline: 3px solid rgba(104, 235, 226, 0.14);
            outline-offset: 3px;
            box-shadow: 0 8px 20px rgba(67, 240, 222, 0.06);
        }

        /* Ensure tooltip hidden on small screens and expand sidebar fully */
        @media (max-width: 760px) {
            section[data-testid="stSidebar"] {
                width: 18.5rem !important;
                min-width: 18.5rem !important;
            }
            section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-tooltip]::after { display: none; }
        }
        </style>

        <script>
        // Attach data-tooltip attributes to sidebar radio labels for CSS tooltips.
        (function() {
            function attachTooltips() {
                const labels = document.querySelectorAll('section[data-testid="stSidebar"] [data-testid="stRadio"] label');
                labels.forEach(l => {
                    try {
                        const text = l.innerText || l.textContent || '';
                        const clean = text.replace(/\n/g, ' ').trim();
                        if (clean) {
                            l.setAttribute('data-tooltip', clean);
                            l.setAttribute('title', clean);
                        }
                    } catch(e) {}
                });
            }
            // run after a short delay to allow Streamlit to render
            setTimeout(attachTooltips, 600);
            // also run on DOM updates every second for a short time
            let runs = 0;
            const iv = setInterval(function(){ attachTooltips(); runs++; if (runs>8) clearInterval(iv); }, 700);
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )

inject_sidebar_enhancements()


# Load model
@st.cache_resource
def load_trained_model():
    try:
        model_path = PROJECT_ROOT / "models" / "churn_model_best.pkl"
        if not model_path.exists():
            st.error("❌ Model not found. Please run `python main.py` first to train the model.")
            return None, None, None, None

        model, scaler, preprocessor, metadata = ModelManager.load_model(str(model_path))
        return model, scaler, preprocessor, metadata
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "churn.csv")


model, scaler, preprocessor, metadata = load_trained_model()

# Sidebar navigation
with st.sidebar:
    st.markdown("### Control Center")
    st.caption("Switch between product views")
    st.caption("A compact navigation rail for dashboard exploration, scoring, and model context.")
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "📈 Dashboard", "🔮 Predict Churn", "📚 Information"]
        ,
        label_visibility="collapsed"
    )

    st.divider()

    if metadata:
        st.markdown("#### Model Snapshot")
        st.caption("Live artifact summary")
        render_snapshot_stack(
            [
                ("Model type", metadata.get('model_type', 'N/A')),
                ("ROC-AUC", f"{metadata.get('metrics', {}).get('ROC-AUC', 0):.4f}"),
                ("Accuracy", f"{metadata.get('metrics', {}).get('Accuracy', 0):.4f}"),
            ]
        )

# PAGE: HOME
if page == "🏠 Home":
    render_hero(
        "Customer Churn Prediction",
        "An enterprise AI/ML analytics dashboard for understanding churn risk, customer behavior, and retention actions.",
        "📊",
        ["Logistic Regression", "AI analytics", "Production dashboard"],
    )

    st.write("")
    performance = metadata.get('metrics', {}) if metadata else {}

    try:
        sample_df = load_sample_data()
        churn_rate = float((sample_df['Churn'] == 'Yes').mean() * 100)
    except Exception:
        sample_df = None
        churn_rate = 0.0

    render_section_header(
        "KPI Snapshot",
        "Core evaluation and portfolio metrics for the current churn model.",
    )

    render_metric_strip([
        ("ROC-AUC", f"{performance.get('ROC-AUC', 0):.4f}", "Primary ranking metric"),
        ("Accuracy", f"{performance.get('Accuracy', 0):.4f}", "Overall correctness"),
        ("Precision", f"{performance.get('Precision', 0):.4f}", "Positive prediction quality"),
        ("Recall", f"{performance.get('Recall', 0):.4f}", "Churn capture rate"),
        ("Customer Risk Score", f"{churn_rate:.1f}%", "Observed churn exposure"),
    ])

    st.write("")
    render_section_header(
        "Executive Overview",
        "Clear operational context for customer retention teams and decision makers.",
    )

    render_info_tiles(
        [
            ("Detect Churn Risk", "Prioritize customers with elevated churn probability before retention windows close."),
            ("Improve Retention", "Focus offers and interventions where business impact is highest."),
            ("Analyze Customer Behavior", "Surface signal patterns across tenure, billing, and service usage."),
            ("Predict Customer Loss", "Support faster, model-backed decisions with measurable confidence."),
        ]
    )

    st.write("")
    render_section_header(
        "Visual Analytics",
        "Widgets summarizing churn distribution, segment variation, and risk patterns.",
    )

    if sample_df is not None:
        chart_left, chart_right = st.columns(2)

        churn_counts = sample_df['Churn'].value_counts()
        churn_fig = go.Figure(data=[go.Pie(
            labels=['No Churn', 'Churn'],
            values=[churn_counts.get('No', 0), churn_counts.get('Yes', 0)],
            hole=0.58,
            sort=False,
            direction="clockwise",
            marker=dict(colors=['#67d29f', '#ff7d7d'], line=dict(color='rgba(255,255,255,0.08)', width=1))
        )])
        churn_fig.update_layout(height=320, margin=dict(l=12, r=12, t=12, b=12), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True)
        with chart_left:
            st.markdown("#### Churn Distribution")
            st.caption("Share of retained versus churned customers.")
            st.plotly_chart(churn_fig, use_container_width=True, config={"displayModeBar": False})

        contract_rate = sample_df.groupby('Contract')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).sort_values(ascending=False)
        contract_fig = go.Figure(data=[go.Bar(x=contract_rate.index, y=contract_rate.values, marker_color=['#7c9cff', '#67d29f', '#f6c76f'])])
        contract_fig.update_layout(height=320, margin=dict(l=12, r=12, t=12, b=12), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, yaxis_title='Churn rate %')
        contract_fig.update_yaxes(gridcolor='rgba(148,163,184,0.16)')
        with chart_right:
            st.markdown("#### Risk by Contract Type")
            st.caption("Churn exposure by subscription structure.")
            st.plotly_chart(contract_fig, use_container_width=True, config={"displayModeBar": False})

        trend_left, trend_right = st.columns(2)
        tenure_bins = pd.cut(sample_df['tenure'], bins=[0, 12, 24, 48, 72], include_lowest=True)
        tenure_trend = sample_df.assign(TenureBand=tenure_bins).groupby('TenureBand')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='ChurnRate')
        trend_fig = go.Figure(data=[go.Scatter(x=[str(x) for x in tenure_trend['TenureBand']], y=tenure_trend['ChurnRate'], mode='lines+markers', line=dict(color='#67f0de', width=3), marker=dict(size=8))])
        trend_fig.update_layout(height=300, margin=dict(l=12, r=12, t=12, b=12), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, yaxis_title='Churn rate %')
        trend_fig.update_yaxes(gridcolor='rgba(148,163,184,0.16)')
        with trend_left:
            st.markdown("#### Prediction Trend")
            st.caption("Churn intensity across customer tenure bands.")
            st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

        monthly_fig = go.Figure()
        monthly_fig.add_trace(go.Box(y=sample_df[sample_df['Churn'] == 'No']['MonthlyCharges'], name='Retained', marker_color='#67d29f', boxmean=True))
        monthly_fig.add_trace(go.Box(y=sample_df[sample_df['Churn'] == 'Yes']['MonthlyCharges'], name='Churn', marker_color='#ff7d7d', boxmean=True))
        monthly_fig.update_layout(height=300, margin=dict(l=12, r=12, t=12, b=12), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True)
        monthly_fig.update_yaxes(gridcolor='rgba(148,163,184,0.16)')
        with trend_right:
            st.markdown("#### Segmentation Risk")
            st.caption("Monthly charge distribution across churn outcomes.")
            st.plotly_chart(monthly_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Sample data is unavailable, so analytics charts could not be rendered.")

    st.write("")
    render_section_header(
        "Model Information",
        "Structured summary of the current saved artifact and deployment posture.",
    )

    model_left, model_right = st.columns([1.25, 0.75])
    with model_left:
        render_info_tiles(
            [
                ("Model Type", metadata.get('model_type', 'N/A') if metadata else 'N/A'),
                ("Dataset Status", "Churn dataset loaded and aligned to the saved artifact."),
                ("Deployment Readiness", "Ready for Streamlit demo and local prediction workflows."),
                ("Prediction Workflow", "Input customer features, score risk, and trigger retention actions."),
            ]
        )
    with model_right:
        render_stat_card("Evaluation Summary", "Logistic Regression", f"Accuracy {performance.get('Accuracy', 0):.4f} • F1 {performance.get('F1 Score', 0):.4f}", compact=False, kind="feature")
        st.write("")
        render_stat_card("Primary Metric", f"ROC-AUC {performance.get('ROC-AUC', 0):.4f}", f"Recall {performance.get('Recall', 0):.4f}", compact=False, kind="feature")

    st.write("")
    render_section_header(
        "Workflow",
        "End-to-end operating flow for the churn analytics experience.",
    )
    render_info_tiles(
        [
            ("1. Upload or input data", "Bring in the customer record or use the sample dataset for analysis."),
            ("2. Process attributes", "Standardize the customer profile and align it to the saved feature schema."),
            ("3. Predict churn", "Generate churn probability and associated risk classification."),
            ("4. Analyze results", "Review metrics, charts, and segment-level insights."),
            ("5. Recommend action", "Apply retention interventions based on the risk tier."),
        ]
    )

    st.write("")
    st.caption("Current best model: Logistic Regression • Production-ready local demo")



# PAGE: DASHBOARD
elif page == "📈 Dashboard":
    render_hero(
        "Dashboard & Statistics",
        "A clean command center for dataset health, churn distribution, and business context.",
        "📈",
        ["Dataset summary", "Visual analytics", "One-click loading"],
    )

    render_section_header(
        "Dashboard Summary",
        "The dashboard loads the sample dataset automatically so the key summary and charts are visible immediately.",
    )

    try:
        with st.spinner("Loading dataset and preparing summary..."):
            df = load_sample_data()
            st.session_state.sample_data = df

        churn_count = int((df['Churn'] == 'Yes').sum())
        churn_rate = churn_count / len(df) * 100
        average_tenure = df['tenure'].mean()

        render_metric_grid([
            ("Total customers", f"{len(df):,}", "Rows loaded from churn.csv"),
            ("Churned", f"{churn_count:,}", "Observed churn outcomes"),
            ("Churn rate", f"{churn_rate:.1f}%", "Share of churned users"),
            ("Average tenure", f"{average_tenure:.0f} months", "Mean customer lifetime"),
        ])

        st.write("")

        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.markdown("#### Churn Distribution")
            st.caption("Share of churned versus retained customers")
            churn_counts = df['Churn'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['No Churn', 'Churn'],
                values=[churn_counts.get('No', 0), churn_counts.get('Yes', 0)],
                hole=0.58,
                sort=False,
                direction="clockwise",
                marker=dict(colors=['#67d29f', '#ff7d7d'], line=dict(color='rgba(255,255,255,0.08)', width=1))
            )])
            apply_chart_theme(fig, height=360)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with chart_right:
            st.markdown("#### Tenure vs Churn")
            st.caption("Tenure profile separated by churn outcome")
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=df[df['Churn'] == 'No']['tenure'],
                name='No Churn',
                marker_color='#67d29f',
                boxmean=True,
            ))
            fig.add_trace(go.Box(
                y=df[df['Churn'] == 'Yes']['tenure'],
                name='Churn',
                marker_color='#ff7d7d',
                boxmean=True,
            ))
            fig.update_yaxes(gridcolor='rgba(148,163,184,0.16)')
            apply_chart_theme(fig, height=360)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.write("")
        render_info_tiles([
            ("Interpretation", "Higher churn density among early-tenure customers signals a retention window worth prioritizing."),
            ("Billing lens", "Monthly billing pressure and contract type are the strongest business-facing signals to watch."),
            ("Usage context", "Use the dashboard to explain customer segments before moving to manual prediction."),
        ])
    except Exception as error:
        st.error(f"Could not load sample data: {error}")


# PAGE: PREDICT CHURN
elif page == "🔮 Predict Churn":
    render_hero(
        "Predict Customer Churn",
        "Enter customer attributes and receive an instant churn probability with a clear recommended action.",
        "🔮",
        ["Probability scoring", "Risk labeling", "Action-oriented output"],
    )
    
    if model is None:
        st.error("❌ Model not loaded. Please train the model first.")
    else:
        render_section_header(
            "Prediction Workspace",
            "The form below is intentionally compact and aligned with the training schema so predictions are stable and fast.",
        )

        form_left, form_right = st.columns(2)
        
        with form_left:
            render_section_header("Customer Profile", "Demographic and household attributes.")
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])

            st.write("")
            render_section_header("Service Usage", "Connectivity and support profile.")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        
        with form_right:
            render_section_header("Account Information", "Tenure and billing signals.")
            tenure = st.number_input("Tenure (months)", 0, 72, 24)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)

            st.write("")
            render_section_header("Additional Services", "Support and billing configuration.")
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        # Make prediction
        if st.button("🔮 Predict Churn", use_container_width=True):
            st.divider()
            
            customer_data = pd.DataFrame({
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
                'gender_Male': [1 if gender == 'Male' else 0],
                'Partner_Yes': [1 if partner == 'Yes' else 0],
                'Dependents_Yes': [1 if dependents == 'Yes' else 0],
                'PhoneService_Yes': [1 if phone_service == 'Yes' else 0],
                'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
                'InternetService_No': [1 if internet_service == 'No' else 0],
                'OnlineSecurity_No internet service': [1 if online_security == 'No internet service' else 0],
                'OnlineSecurity_Yes': [1 if online_security == 'Yes' else 0],
                'TechSupport_No internet service': [1 if tech_support == 'No internet service' else 0],
                'TechSupport_Yes': [1 if tech_support == 'Yes' else 0],
                'StreamingTV_No internet service': [1 if streaming_tv == 'No internet service' else 0],
                'StreamingTV_Yes': [1 if streaming_tv == 'Yes' else 0],
                'Contract_One year': [1 if contract == 'One year' else 0],
                'Contract_Two year': [1 if contract == 'Two year' else 0],
                'PaperlessBilling_Yes': [1 if paperless_billing == 'Yes' else 0],
            })
            
            try:
                # Align the manual input with the full training feature set.
                feature_names = preprocessor.get('feature_names', []) if preprocessor else []
                if feature_names:
                    aligned_data = pd.DataFrame(0, index=[0], columns=feature_names)
                    for column, value in customer_data.iloc[0].items():
                        if column in aligned_data.columns:
                            aligned_data.at[0, column] = value
                    customer_scaled = scaler.transform(aligned_data)
                else:
                    customer_scaled = scaler.transform(customer_data)

                prediction = model.predict(customer_scaled)[0]
                probability = model.predict_proba(customer_scaled)[0][1]
                
                # Determine risk level
                if probability >= 0.7:
                    risk_level = "HIGH"
                    risk_class = "danger"
                elif probability >= 0.4:
                    risk_level = "MEDIUM"
                    risk_class = "warning"
                else:
                    risk_level = "LOW"
                    risk_class = "success"
                
                # Display results
                render_section_header("Prediction Result", "Clean output for recruiter-ready presentation.")
                render_metric_grid([
                    ("Churn probability", f"{probability:.1%}", f"Risk tier: {risk_level}"),
                    ("No-churn probability", f"{1-probability:.1%}", "Complementary confidence"),
                    ("Prediction", "WILL CHURN" if prediction == 1 else "WILL STAY", "Binary classification result"),
                    ("Risk band", risk_level, "Actionable customer grouping"),
                ])

                st.write("")
                render_section_header("Recommendation", "Suggested next action based on the risk band.")
                if probability >= 0.7:
                    st.error(
                        """
                        **High risk detected** - This customer is likely to churn.

                        Recommended actions:
                        - Contact proactively with a retention offer
                        - Offer a contract upgrade incentive
                        - Provide a service upgrade discount
                        - Act within 7 days for best results
                        """
                    )
                elif probability >= 0.4:
                    st.warning(
                        """
                        **Medium risk detected** - Monitor this customer closely.

                        Recommended actions:
                        - Add to monitoring list
                        - Send a targeted offer email
                        - Highlight service benefits
                        """
                    )
                else:
                    st.success(
                        """
                        **Low risk detected** - Customer satisfaction is likely high.

                        Recommended actions:
                        - Continue regular service
                        - Request a satisfaction survey
                        - Encourage referrals
                        """
                    )
                    
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")


# PAGE: INFORMATION
elif page == "📚 Information":
    render_hero(
        "Project Information",
        "A concise product brief covering the dataset, model, and the customer risk signals the application uses.",
        "📚",
        ["Model details", "Feature summary", "Business context"],
    )
    
    tab1, tab2, tab3 = st.tabs(["About", "Model Details", "Features"])
    
    with tab1:
        render_section_header(
            "About this project",
            "Predict which telecom customers are likely to churn and give the operations team a clean, immediate way to focus retention effort.",
        )

        render_info_tiles(
            [
                ("Dataset", "7,043 telecom customers with demographic, service, and billing information."),
                ("Best model", "Logistic Regression selected by ROC-AUC after balancing the training set with SMOTE."),
                ("Business value", "Helps reduce acquisition spend by identifying high-risk customers before they leave."),
            ]
        )
    
    with tab2:
        render_section_header("Model Performance", "A compact chart view of the current evaluation snapshot.")
        
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
            'Score': [0.8152, 0.6654, 0.6146, 0.6391, 0.8363]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = go.Figure(data=[
            go.Bar(x=metrics_df['Metric'], y=metrics_df['Score'], marker_color=['#7c9cff', '#67d29f', '#f6c76f', '#78dcca', '#9f7aea'])
        ])
        fig.update_layout(title=dict(text="Model Performance Metrics", font=dict(color='#e6eefc', size=20)), yaxis_title="Score")
        fig.update_yaxes(gridcolor='rgba(148,163,184,0.16)', range=[0, 1])
        apply_chart_theme(fig, height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### Model comparison")
        comparison_df = pd.DataFrame(
            [
                ["Logistic Regression", "0.8363", "Best"],
                ["Gradient Boosting", "0.8310", "Very Good"],
                ["AdaBoost", "0.8250", "Good"],
                ["Random Forest", "0.8230", "Good"],
                ["XGBoost", "0.8150", "Good"],
                ["Decision Tree", "0.6551", "Baseline"],
            ],
            columns=["Model", "ROC-AUC", "Status"],
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with tab3:
        render_section_header("Top churn predictors", "The most important business signals surfaced by the project.")
        render_info_tiles(
            [
                ("Contract type", "Month-to-month customers churn significantly more often than long-term subscribers."),
                ("Tenure", "Early lifecycle customers need the most retention attention."),
                ("Monthly charges", "Higher billing correlates with churn risk and price sensitivity."),
            ]
        )
        st.write("")
        st.info("Operational takeaway: prioritize customers on short contracts, with low tenure, and higher monthly charges.")

# Footer
st.divider()
st.caption("Built for customer retention analytics • Streamlit + scikit-learn")
