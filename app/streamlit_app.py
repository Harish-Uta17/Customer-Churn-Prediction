"""
Streamlit App - Customer Churn Prediction

Interactive web application for predicting customer churn.

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
            --bg: #0b1020;
            --panel: rgba(15, 23, 42, 0.86);
            --panel-strong: rgba(17, 24, 39, 0.95);
            --panel-soft: rgba(30, 41, 59, 0.72);
            --border: rgba(148, 163, 184, 0.16);
            --text: #e5eefc;
            --muted: #93a4c3;
            --accent: #78dcca;
            --accent-2: #7c9cff;
            --accent-3: #f6c76f;
            --danger: #ff7d7d;
            --success: #67d29f;
            --shadow: 0 24px 60px rgba(0, 0, 0, 0.34);
            --radius-xl: 24px;
            --radius-lg: 18px;
            --radius-md: 14px;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(124, 156, 255, 0.12), transparent 30%),
                radial-gradient(circle at top right, rgba(120, 220, 202, 0.10), transparent 26%),
                linear-gradient(180deg, #090d18 0%, #0b1020 36%, #0b1020 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1400px;
        }

        h1, h2, h3, h4, p, label, div {
            color: var(--text);
        }

        .app-shell {
            border: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.90), rgba(10, 15, 30, 0.92));
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow);
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 30px;
            background:
                linear-gradient(135deg, rgba(124, 156, 255, 0.15), rgba(120, 220, 202, 0.08)),
                linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.78));
            box-shadow: var(--shadow);
            padding: 1.6rem 1.7rem;
            margin: 0 0 1.25rem 0;
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: auto -80px -120px auto;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(120, 220, 202, 0.25), transparent 66%);
            pointer-events: none;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            background: rgba(124, 156, 255, 0.12);
            border: 1px solid rgba(124, 156, 255, 0.18);
            color: var(--accent-2);
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
        }

        .hero-title {
            margin-top: 0.9rem;
            margin-bottom: 0.45rem;
            font-size: clamp(2rem, 3vw, 3.2rem);
            font-weight: 800;
            line-height: 1.02;
        }

        .hero-subtitle {
            max-width: 760px;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.65;
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
            background: rgba(148, 163, 184, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.14);
            color: #d9e4f7;
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

        .card {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.94), rgba(15, 23, 42, 0.84));
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.05rem 1.1rem;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
            gap: 1rem;
        }

        .metric-card {
            position: relative;
            overflow: hidden;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 42, 0.84));
            border: 1px solid var(--border);
            padding: 1rem 1.05rem;
            min-height: 128px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.24);
        }

        .metric-card::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, var(--accent-2), var(--accent));
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .metric-value {
            margin-top: 0.6rem;
            font-size: clamp(1.8rem, 2.8vw, 2.8rem);
            font-weight: 800;
            line-height: 1;
        }

        .metric-note {
            margin-top: 0.6rem;
            color: var(--muted);
            font-size: 0.88rem;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
        }

        .metric-pill {
            border-radius: 16px;
            padding: 0.8rem 0.85rem;
            background: rgba(17, 24, 39, 0.82);
            border: 1px solid var(--border);
        }

        .metric-pill-label {
            color: var(--muted);
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .metric-pill-value {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.15;
            white-space: nowrap;
        }

        .metric-pill-note {
            color: var(--muted);
            font-size: 0.8rem;
            margin-top: 0.25rem;
        }

        .snapshot-stack {
            display: flex;
            flex-direction: column;
            gap: 0.7rem;
        }

        .snapshot-card {
            border-radius: 16px;
            padding: 0.8rem 0.85rem;
            background: rgba(17, 24, 39, 0.72);
            border: 1px solid var(--border);
        }

        .snapshot-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }

        .snapshot-value {
            color: var(--text);
            font-size: 1.02rem;
            font-weight: 800;
            line-height: 1.2;
            word-break: break-word;
        }

        .accent-card {
            border-radius: 22px;
            padding: 1.1rem 1.15rem;
            border: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(120, 220, 202, 0.12), rgba(124, 156, 255, 0.06));
        }

        .input-card .stSelectbox, .input-card .stNumberInput {
            margin-bottom: 0.4rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(124, 156, 255, 0.28);
            background: linear-gradient(135deg, rgba(124, 156, 255, 0.22), rgba(120, 220, 202, 0.16));
            color: white;
            font-weight: 700;
            padding: 0.7rem 1rem;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(120, 220, 202, 0.44);
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.26);
        }

        .stMetric {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }

        .stMetric [data-testid="metric-container"] {
            background: transparent !important;
            border: none !important;
        }

        .stMetric label,
        .stMetric [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
        }

        .stMetric [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-weight: 800 !important;
        }

        .stInfo, .stSuccess, .stWarning, .stError {
            border-radius: 18px;
            border: 1px solid var(--border);
        }

        .subtle-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.28), transparent);
            margin: 1.25rem 0;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
        }

        .info-tile {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.94), rgba(15, 23, 42, 0.82));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.05rem;
        }

        .info-tile h4 {
            margin: 0 0 0.35rem;
            font-size: 1rem;
            font-weight: 800;
        }

        .info-tile p {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
        }

        @media (max-width: 1100px) {
            .metric-grid, .info-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero {
                padding: 1.1rem 1rem;
                border-radius: 24px;
            }

            .metric-grid, .info-grid {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 1.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, icon: str, chips: list[str] | None = None) -> None:
    chip_html = ""
    if chips:
        chip_html = "<div class='chip-row'>" + "".join([f"<span class='chip'>{chip}</span>" for chip in chips]) + "</div>"

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{icon} Customer churn intelligence</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metric_strip(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class='metric-pill'>
                    <div class='metric-pill-label'>{label}</div>
                    <div class='metric-pill-value'>{value}</div>
                    <div class='metric-pill-note'>{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_snapshot_stack(items: list[tuple[str, str]]) -> None:
    for label, value in items:
        st.markdown(
            f"""
            <div class='snapshot-card'>
                <div class='snapshot-label'>{label}</div>
                <div class='snapshot-value'>{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_info_tiles(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (title, body) in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="info-tile">
                    <h4>{title}</h4>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


inject_styles()


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
    st.markdown("## Navigation")
    st.caption("Switch between product views")
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "📈 Dashboard", "🔮 Predict Churn", "📚 Information"]
    )

    st.markdown("<div class='subtle-divider'></div>", unsafe_allow_html=True)

    if metadata:
        st.markdown("#### Model Snapshot")
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
        "A polished ML workspace for exploring churn drivers, evaluating customer risk, and making retention decisions with confidence.",
        "📊",
        ["Logistic Regression", "ROC-AUC driven", "Streamlit UI", "Responsive layout"],
    )

    col1, col2 = st.columns([1.7, 1])
    
    with col1:
        st.markdown("<div class='section-title'>Executive Overview</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>The application combines a trained churn model with a modern decision dashboard. It is designed for fast operational use, not notebook-style exploration.</div>",
            unsafe_allow_html=True,
        )

        render_info_tiles(
            [
                ("Detect churn risk", "Rank customers by likely churn probability and act before retention windows close."),
                ("Explain outcomes", "See concise model performance and customer-level recommendations in one place."),
                ("Operate quickly", "Use a compact flow that works on desktop, laptop, and smaller screens."),
            ]
        )

        st.markdown("<div class='subtle-divider'></div>", unsafe_allow_html=True)
        st.markdown("### Workflow")
        st.markdown(
            """
            <div class='card'>
            1. Load the saved model and training schema.
            <br>2. Inspect dashboard indicators or sample data.
            <br>3. Enter customer attributes in Predict Churn.
            <br>4. Review risk, probability, and recommended actions.
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        if metadata:
            st.markdown("<div class='section-title'>Model Card</div>", unsafe_allow_html=True)
            performance = metadata.get('metrics', {})
            if performance:
                st.markdown(
                    """
                    <div class='card'>
                        <div class='metric-label'>Model type</div>
                        <div class='metric-value' style='font-size:1.45rem;'>Logistic Regression</div>
                        <div class='metric-note'>Current best model chosen by ROC-AUC on the validation set.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                render_metric_strip([
                    ("ROC-AUC", f"{performance.get('ROC-AUC', 0):.4f}", "Primary ranking metric"),
                    ("Accuracy", f"{performance.get('Accuracy', 0):.4f}", "Overall correctness"),
                    ("Precision", f"{performance.get('Precision', 0):.4f}", "Positive prediction quality"),
                ])

                st.markdown(
                    f"<div class='metric-note' style='margin-top:0.7rem;'>Recall {performance.get('Recall', 0):.4f} • Churn capture rate</div>",
                    unsafe_allow_html=True,
                )


# PAGE: DASHBOARD
elif page == "📈 Dashboard":
    render_hero(
        "Dashboard & Statistics",
        "A clean command center for dataset health, churn distribution, and business context.",
        "📈",
        ["Dataset summary", "Visual analytics", "One-click loading"],
    )
    
    st.markdown(
        "<div class='section-copy'>The dashboard loads the sample dataset automatically so the key summary and charts are visible immediately.</div>",
        unsafe_allow_html=True,
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

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.markdown("#### Churn Distribution")
            churn_counts = df['Churn'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['No Churn', 'Churn'],
                values=[churn_counts.get('No', 0), churn_counts.get('Yes', 0)],
                hole=0.58,
                sort=False,
                direction="clockwise",
                marker=dict(colors=['#67d29f', '#ff7d7d'], line=dict(color='rgba(255,255,255,0.08)', width=1))
            )])
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=-0.18, x=0.05),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=360,
                font=dict(color='#e5eefc'),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with chart_right:
            st.markdown("#### Tenure vs Churn")
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
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=360,
                font=dict(color='#e5eefc'),
                yaxis=dict(gridcolor='rgba(148,163,184,0.16)'),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
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
        st.markdown("<div class='section-copy'>The form below is intentionally compact and aligned with the training schema so predictions are stable and fast.</div>", unsafe_allow_html=True)

        form_left, form_right = st.columns([1.15, 1.15])
        
        with form_left:
            st.markdown("#### Demographics", unsafe_allow_html=True)
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            
            st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
            st.markdown("#### Services", unsafe_allow_html=True)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        
        with form_right:
            st.markdown("#### Account Information", unsafe_allow_html=True)
            tenure = st.number_input("Tenure (months)", 0, 72, 24)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)
            
            st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
            st.markdown("#### Other Services", unsafe_allow_html=True)
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        # Make prediction
        if st.button("🔮 Predict Churn", use_container_width=True):
            st.markdown("<div class='subtle-divider'></div>", unsafe_allow_html=True)
            
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
                render_metric_grid([
                    ("Churn probability", f"{probability:.1%}", f"Risk tier: {risk_level}"),
                    ("No-churn probability", f"{1-probability:.1%}", "Complementary confidence"),
                    ("Prediction", "WILL CHURN" if prediction == 1 else "WILL STAY", "Binary classification result"),
                    ("Risk band", risk_level, "Actionable customer grouping"),
                ])

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Recommendation</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-title'>About this project</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-copy'>Predict which telecom customers are likely to churn and give the operations team a clean, immediate way to focus retention effort.</div>", unsafe_allow_html=True)

        render_info_tiles(
            [
                ("Dataset", "7,043 telecom customers with demographic, service, and billing information."),
                ("Best model", "Logistic Regression selected by ROC-AUC after balancing the training set with SMOTE."),
                ("Business value", "Helps reduce acquisition spend by identifying high-risk customers before they leave."),
            ]
        )
    
    with tab2:
        st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
        
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
            'Score': [0.8152, 0.6654, 0.6146, 0.6391, 0.8363]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = go.Figure(data=[
            go.Bar(x=metrics_df['Metric'], y=metrics_df['Score'], marker_color=['#7c9cff', '#67d29f', '#f6c76f', '#78dcca', '#9f7aea'])
        ])
        fig.update_layout(
            title=dict(text="Model Performance Metrics", font=dict(color='#e5eefc', size=20)),
            yaxis_title="Score",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5eefc'),
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
            yaxis=dict(gridcolor='rgba(148,163,184,0.16)')
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Model comparison", unsafe_allow_html=True)
        st.markdown(
            """
            | Model | ROC-AUC | Status |
            |-------|---------|--------|
            | **Logistic Regression** | **0.8363** | Best |
            | Gradient Boosting | 0.8310 | Very Good |
            | AdaBoost | 0.8250 | Good |
            | Random Forest | 0.8230 | Good |
            | XGBoost | 0.8150 | Good |
            | Decision Tree | 0.6551 | Baseline |
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='section-title'>Top churn predictors</div>", unsafe_allow_html=True)
        render_info_tiles(
            [
                ("Contract type", "Month-to-month customers churn significantly more often than long-term subscribers."),
                ("Tenure", "Early lifecycle customers need the most retention attention."),
                ("Monthly charges", "Higher billing correlates with churn risk and price sensitivity."),
            ]
        )
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='card'>
            <strong>Operational takeaway:</strong> prioritize customers on short contracts, with low tenure, and higher monthly charges.
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown("<div class='subtle-divider'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; color:#93a4c3; padding: 0.5rem 0 0.25rem;'>
        Built for customer retention analytics • Streamlit + scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
