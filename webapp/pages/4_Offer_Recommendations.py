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

from utils import load_customer_data

st.set_page_config(page_title="Offer Recommendations", layout="wide")

# Load custom CSS
with open(Path(__file__).parent.parent / 'assets' / 'style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>Customer 360 Analytics <span style="color:#00E5FF;">| Offer Recommendations</span></h1>
    <hr>
</div>
""", unsafe_allow_html=True)

df = load_customer_data()
if df.empty:
    st.stop()

# Offer distribution
st.subheader("Recommended Offers Distribution")
offer_counts = df['RecommendedOffer'].value_counts().reset_index()
offer_counts.columns = ['Offer', 'Count']
fig = px.bar(
    offer_counts,
    x='Count',
    y='Offer',
    orientation='h',
    title='Number of Customers per Offer',
    color='Count',
    color_continuous_scale='Purples'
)
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, use_container_width=True)

# Select offer to view customers
st.subheader("Customers by Offer")
selected_offer = st.selectbox("Select an offer to see customers", df['RecommendedOffer'].unique())
offer_customers = df[df['RecommendedOffer'] == selected_offer][['CustomerID', 'Segment', 'ChurnProb', 'HistoricalLTV']]
st.dataframe(
    offer_customers,
    use_container_width=True,
    column_config={
        "ChurnProb": st.column_config.ProgressColumn("Churn Risk", format="%.1f%%", min_value=0, max_value=1),
        "HistoricalLTV": st.column_config.NumberColumn("Historical LTV", format="$%.0f")
    }
)

# Footer
st.markdown("""
<div class="footer">
    <hr>
    <p>© 2026 Customer 360 Analytics. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)