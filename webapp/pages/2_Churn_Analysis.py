import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

import plotly.graph_objects as go

# ----------------------
# Plotly theme: dark-glass (matches custom CSS)
# ----------------------
_CUSTOM_PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#E8EEFF",
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)"),
        colorway=[
            "#00E5FF", "#FF3D81", "#7C4DFF", "#00E676",
            "#FFD54F", "#FF6D00", "#64FFDA", "#B388FF",
        ],
    )
)
pio.templates["customer360_dark"] = _CUSTOM_PLOTLY_TEMPLATE
pio.templates.default = "customer360_dark"

from utils import load_customer_data, load_churn_model, get_feature_columns, metric_card

st.set_page_config(page_title="Churn Analysis", layout="wide")

# Load custom CSS
with open(Path(__file__).parent.parent / 'assets' / 'style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>Customer 360 Analytics <span style="color:#00E5FF;">| Churn Analysis</span></h1>
    <hr>
</div>
""", unsafe_allow_html=True)

df = load_customer_data()
model = load_churn_model()
if df.empty or model is None:
    st.stop()

# Sidebar filters
with st.sidebar:
    st.markdown("## ⚠️ Churn Filters")
    churn_threshold = st.slider("High-risk threshold", 0.0, 1.0, 0.7, 0.05)

# Churn probability distribution
st.subheader("Churn Probability Distribution")
fig_hist = px.histogram(
    df,
    x='ChurnProb',
    nbins=50,
    title='Distribution of Churn Probabilities',
    color_discrete_sequence=['#00E5FF']
)
fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_hist, use_container_width=True)

# High-risk customers
st.subheader(f"🔴 High-Risk Customers (Churn Probability > {churn_threshold:.0%})")
high_risk = df[df['ChurnProb'] > churn_threshold].sort_values('ChurnProb', ascending=False)
st.dataframe(
    high_risk[['CustomerID', 'Segment', 'Recency', 'Frequency', 'ChurnProb']],
    use_container_width=True,
    column_config={
        "ChurnProb": st.column_config.ProgressColumn("Churn Risk", format="%.1f%%", min_value=0, max_value=1)
    }
)

# Feature importance
st.subheader("Feature Importance (Random Forest)")
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=get_feature_columns()).sort_values(ascending=True)
fig_imp = px.bar(
    feat_imp,
    x=feat_imp.values,
    y=feat_imp.index,
    orientation='h',
    title='What Drives Churn?',
    labels={'x': 'Importance', 'y': ''},
    color_discrete_sequence=['#FF3D81']
)
fig_imp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_imp, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <hr>
    <p>© 2026 Customer 360 Analytics. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)