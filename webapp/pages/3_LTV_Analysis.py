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

st.set_page_config(page_title="LTV Analysis", layout="wide")

# Load custom CSS
with open(Path(__file__).parent.parent / 'assets' / 'style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>Customer 360 Analytics <span style="color:#00E5FF;">| LTV Analysis</span></h1>
    <hr>
</div>
""", unsafe_allow_html=True)

df = load_customer_data()
if df.empty:
    st.stop()

# LTV distributions
col1, col2 = st.columns(2)
with col1:
    fig_hist1 = px.histogram(
        df,
        x='HistoricalLTV',
        nbins=50,
        title='Historical LTV Distribution',
        color_discrete_sequence=['#00E5FF']
    )
    fig_hist1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hist1, use_container_width=True)
with col2:
    fig_hist2 = px.histogram(
        df,
        x='PredictedLTV_Next6Months',
        nbins=50,
        title='Predicted LTV (Next 6 Months)',
        color_discrete_sequence=['#00E676']
    )
    fig_hist2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hist2, use_container_width=True)

# Top customers
st.subheader("🏆 Top 20 Customers by Historical LTV")
top_ltv = df.nlargest(20, 'HistoricalLTV')[['CustomerID', 'Segment', 'HistoricalLTV', 'PredictedLTV_Next6Months']]
st.dataframe(
    top_ltv,
    use_container_width=True,
    column_config={
        "HistoricalLTV": st.column_config.NumberColumn("Historical LTV", format="$%.0f"),
        "PredictedLTV_Next6Months": st.column_config.NumberColumn("Predicted LTV", format="$%.0f")
    }
)

# LTV by segment
st.subheader("LTV by Segment")
fig_box = px.box(
    df,
    x='Segment',
    y='HistoricalLTV',
    title='Historical LTV Distribution by Segment',
    color='Segment',
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_box.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <hr>
    <p>© 2026 Customer 360 Analytics. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)