import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

@st.cache_data(ttl=3600)
def load_customer_data():
    """Load final customer dataset with offers."""
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'customers_with_offers.csv'
    if not data_path.exists():
        st.error(f"❌ Data file not found at {data_path}. Please run the pipeline first.")
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    df['CustomerID'] = df['CustomerID'].astype(str)
    return df

@st.cache_resource
def load_churn_model():
    model_path = Path(__file__).parent.parent / 'models' / 'churn_model_rf.pkl'
    if not model_path.exists():
        st.error("❌ Churn model not found.")
        return None
    return joblib.load(model_path)

@st.cache_resource
def load_ltv_model():
    model_path = Path(__file__).parent.parent / 'models' / 'ltv_model_rf.pkl'
    if not model_path.exists():
        st.warning("⚠️ LTV model not found. Predictive LTV will be unavailable.")
        return None
    return joblib.load(model_path)

def get_feature_columns():
    return ['Recency', 'Frequency', 'Monetary', 'TenureDays', 'AvgOrderValue']

def metric_card(title, value, delta=None, delta_color="normal", help_text=""):
    """Generate HTML for a metric card."""
    delta_class = "positive" if delta and delta.startswith('+') else "negative" if delta and delta.startswith('-') else ""
    delta_html = f'<p class="delta {delta_class}">{delta}</p>' if delta else ''
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <p class="value">{value}</p>
        {delta_html}
        <small>{help_text}</small>
    </div>
    """