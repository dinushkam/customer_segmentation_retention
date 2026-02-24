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

st.set_page_config(page_title="RFM Segments", layout="wide")

# Load custom CSS
with open(Path(__file__).parent.parent / 'assets' / 'style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>Customer 360 Analytics <span style="color:#00E5FF;">| RFM Segments</span></h1>
    <hr>
</div>
""", unsafe_allow_html=True)

df = load_customer_data()
if df.empty:
    st.stop()

# Sidebar filters
with st.sidebar:
    st.markdown("## 🎯 RFM Filters")
    selected_scores = st.multiselect(
        "RFM Score (3-digit)",
        options=df['RFM_Score'].unique(),
        default=[]
    )

filtered_df = df if not selected_scores else df[df['RFM_Score'].isin(selected_scores)]

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    fig_hist = px.histogram(
        filtered_df,
        x='RFM_Score',
        color='Segment',
        title='RFM Score Distribution',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        barmode='group'
    )
    fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("Segment Profiles")
    profiles = filtered_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
    st.dataframe(profiles.style.background_gradient(cmap='Purples'), use_container_width=True)

st.subheader("3D View of RFM Space")
fig_3d = px.scatter_3d(
    filtered_df,
    x='Recency',
    y='Frequency',
    z='Monetary',
    color='Segment',
    hover_data=['CustomerID'],
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title='RFM Space'
)
fig_3d.update_layout(scene=dict(xaxis_title='Recency', yaxis_title='Frequency', zaxis_title='Monetary'))
st.plotly_chart(fig_3d, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <hr>
    <p>© 2026 Customer 360 Analytics. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)