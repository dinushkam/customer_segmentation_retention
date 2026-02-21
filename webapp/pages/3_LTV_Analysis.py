import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import streamlit as st
import plotly.express as px
from utils import load_customer_data

st.set_page_config(page_title="LTV Analysis", layout="wide")
st.title("💰 Customer Lifetime Value")

df = load_customer_data()

# LTV distributions
col1, col2 = st.columns(2)
with col1:
    fig_hist = px.histogram(df, x='HistoricalLTV', nbins=50, title='Historical LTV Distribution')
    st.plotly_chart(fig_hist, use_container_width=True)
with col2:
    fig_hist2 = px.histogram(df, x='PredictedLTV_Next6Months', nbins=50, title='Predicted LTV (Next 6 Months)')
    st.plotly_chart(fig_hist2, use_container_width=True)

# Top customers by LTV
st.subheader("Top 20 Customers by Historical LTV")
top_ltv = df.nlargest(20, 'HistoricalLTV')[['CustomerID','Segment','HistoricalLTV','PredictedLTV_Next6Months']]
st.dataframe(top_ltv)

# LTV by segment
st.subheader("Average LTV by Segment")
avg_ltv = df.groupby('Segment')[['HistoricalLTV','PredictedLTV_Next6Months']].mean().round(0)
st.dataframe(avg_ltv)

fig_box = px.box(df, x='Segment', y='HistoricalLTV', title='Historical LTV by Segment')
st.plotly_chart(fig_box, use_container_width=True)