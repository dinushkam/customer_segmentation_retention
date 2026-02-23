import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import yaml
from utils import load_customer_data, metric_card

st.set_page_config(page_title="Customer 360 Analytics", layout="wide")

# Load config
with open(Path(__file__).parent / 'config.yaml') as f:
    config = yaml.safe_load(f)

# Load custom CSS
with open(Path(__file__).parent / 'assets' / 'style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



# ======================
# MAIN HEADER (Decent & Bold)
# ======================
st.markdown(f"""
<div class="header" style="margin-bottom: 1rem;">
    <h1 style="font-size: 3rem; font-weight: 700; color: {config['dashboard']['theme']['primary_color']};">📊 {config['dashboard']['title']}</h1>
    <h3 style="color: #4a4a4a; font-weight: 400;">Empower your marketing decisions with data‑driven customer insights.</h3>
    <hr style="border: 2px solid {config['dashboard']['theme']['primary_color']}; opacity: 0.3;">
</div>
""", unsafe_allow_html=True)

# Load data
df = load_customer_data()
if df.empty:
    st.stop()

# ======================
# SIDEBAR WITH DECENT HEADING
# ======================

with st.sidebar:
    # Logo or title
    assets_dir = Path(__file__).parent / 'assets'
    logo_path = assets_dir / 'logo.jpg'

    if logo_path.exists():
        st.image(str(logo_path), width=200)
    else:
        st.warning("Logo not found. Using placeholder.")
        st.image("https://via.placeholder.com/200x80?text=Your+Logo", width=200)

    st.markdown("---")
    st.markdown("## 🎛️ Dashboard Controls")
    st.markdown("Use the filters below to customize the view.")

    with st.form(key="filter_form"):
        st.markdown("#### 🔍 Filters")
        segments = st.multiselect(
            "Customer Segment",
            options=df['Segment'].unique(),
            default=df['Segment'].unique()
        )
        col1, col2 = st.columns(2)
        with col1:
            min_ltv = st.number_input("Min LTV ($)", value=float(df['HistoricalLTV'].min()))
        with col2:
            max_ltv = st.number_input("Max LTV ($)", value=float(df['HistoricalLTV'].max()))
        min_churn, max_churn = st.slider(
            "Churn Probability",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0)
        )
        st.markdown("---")
        apply_filters = st.form_submit_button("Apply Filters", type="primary")

    st.markdown("---")


# Apply filters
filtered_df = df[
    (df['Segment'].isin(segments)) &
    (df['HistoricalLTV'].between(min_ltv, max_ltv)) &
    (df['ChurnProb'].between(min_churn, max_churn))
]

if filtered_df.empty:
    st.warning("⚠️ No customers match the selected filters. Adjust filters.")
    st.stop()

# Key Metrics (as before)
st.subheader("📈 Key Performance Indicators")
cols = st.columns(4)

with cols[0]:
    st.markdown(metric_card(
        "Total Customers",
        f"{len(filtered_df):,}",
        help_text="Active customers in selection"
    ), unsafe_allow_html=True)

with cols[1]:
    avg_churn = filtered_df['ChurnProb'].mean()
    overall_churn = df['ChurnProb'].mean()
    delta_churn = f"{avg_churn - overall_churn:+.2%}"
    st.markdown(metric_card(
        "Avg Churn Probability",
        f"{avg_churn:.2%}",
        delta=delta_churn,
        help_text="vs overall average"
    ), unsafe_allow_html=True)

with cols[2]:
    avg_ltv = filtered_df['HistoricalLTV'].mean()
    overall_ltv = df['HistoricalLTV'].mean()
    delta_ltv = f"${avg_ltv - overall_ltv:+,.0f}"
    st.markdown(metric_card(
        "Avg Historical LTV",
        f"${avg_ltv:,.0f}",
        delta=delta_ltv,
        help_text="vs overall average"
    ), unsafe_allow_html=True)

with cols[3]:
    avg_pred = filtered_df['PredictedLTV_Next6Months'].mean()
    overall_pred = df['PredictedLTV_Next6Months'].mean()
    delta_pred = f"${avg_pred - overall_pred:+,.0f}"
    st.markdown(metric_card(
        "Avg Predicted LTV (6m)",
        f"${avg_pred:,.0f}",
        delta=delta_pred,
        help_text="vs overall average"
    ), unsafe_allow_html=True)

# Tabs (unchanged)
tab1, tab2, tab3 = st.tabs(["📊 Segment Overview", "📈 Risk & Value", "📋 Raw Data"])

with tab1:
    col_left, col_right = st.columns([1, 2])
    with col_left:
        fig_seg = px.pie(
            filtered_df,
            names='Segment',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Segment Distribution"
        )
        fig_seg.update_layout(showlegend=False)
        st.plotly_chart(fig_seg, use_container_width=True)
    with col_right:
        st.subheader("Segment Profiles")
        profiles = filtered_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
        st.dataframe(profiles.style.background_gradient(cmap='Purples'), use_container_width=True)

with tab2:
    fig_scatter = px.scatter(
        filtered_df,
        x='ChurnProb',
        y='HistoricalLTV',
        color='Segment',
        size='PredictedLTV_Next6Months',
        hover_data=['CustomerID'],
        title='Customers by Churn Risk and Lifetime Value',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={'ChurnProb': 'Churn Probability', 'HistoricalLTV': 'Historical LTV ($)'}
    )
    fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.dataframe(filtered_df, use_container_width=True)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download data as CSV",
        data=csv,
        file_name='filtered_customers.csv',
        mime='text/csv',
    )

# Footer
st.markdown("""
<div class="footer">
    <hr>
    <p>© 2026 Customer 360 Analytics. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)