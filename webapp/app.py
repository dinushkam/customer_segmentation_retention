import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
from utils import load_customer_data

st.set_page_config(page_title="Customer 360 Analytics", layout="wide")

# Load config
with open(Path(__file__).parent / 'config.yaml') as f:
    config = yaml.safe_load(f)

# Custom CSS
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        color: {config['dashboard']['theme']['primary_color']};
        font-weight: 600;
        margin-bottom: 1rem;
    }}
    .metric-card {{
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<p class="main-header">{config["dashboard"]["title"]}</p>', unsafe_allow_html=True)
st.markdown("Empower your marketing decisions with data-driven customer insights.")

df = load_customer_data()
if df.empty:
    st.stop()

# Sidebar filters
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Your+Logo", width=150)
    st.markdown("## Filters")
    segments = df['Segment'].unique().tolist()
    selected_segments = st.multiselect("Segment", segments, default=segments)
    min_ltv, max_ltv = st.slider("Historical LTV ($)",
                                  float(df['HistoricalLTV'].min()),
                                  float(df['HistoricalLTV'].max()),
                                  (float(df['HistoricalLTV'].min()), float(df['HistoricalLTV'].max())))
    min_churn, max_churn = st.slider("Churn Probability", 0.0, 1.0, (0.0, 1.0))

filtered_df = df[
    (df['Segment'].isin(selected_segments)) &
    (df['HistoricalLTV'].between(min_ltv, max_ltv)) &
    (df['ChurnProb'].between(min_churn, max_churn))
]
if filtered_df.empty:
    st.warning("No customers match filters.")
    st.stop()

# KPIs
st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", f"{len(filtered_df):,}")
with col2:
    avg = filtered_df['ChurnProb'].mean()
    overall = df['ChurnProb'].mean()
    st.metric("Avg Churn Probability", f"{avg:.2%}", f"{avg-overall:+.2%}", delta_color="inverse")
with col3:
    avg = filtered_df['HistoricalLTV'].mean()
    overall = df['HistoricalLTV'].mean()
    st.metric("Avg Historical LTV", f"${avg:,.0f}", f"${avg-overall:+,.0f}")
with col4:
    avg = filtered_df['PredictedLTV_Next6Months'].mean()
    overall = df['PredictedLTV_Next6Months'].mean()
    st.metric("Avg Predicted LTV (6m)", f"${avg:,.0f}", f"${avg-overall:+,.0f}")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Segment Overview", "📈 Risk & Value", "📋 Raw Data"])
with tab1:
    col_left, col_right = st.columns([1,2])
    with col_left:
        fig = px.pie(filtered_df, names='Segment', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        profiles = filtered_df.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(1)
        st.dataframe(profiles.style.background_gradient(cmap='Blues'))
with tab2:
    fig = px.scatter(filtered_df, x='ChurnProb', y='HistoricalLTV', color='Segment',
                     size='PredictedLTV_Next6Months', hover_data=['CustomerID'])
    st.plotly_chart(fig, use_container_width=True)
with tab3:
    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "filtered_customers.csv", "text/csv")