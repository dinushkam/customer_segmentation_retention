import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import streamlit as st
import plotly.express as px
from utils import load_customer_data

st.set_page_config(page_title="Offer Recommendations", layout="wide")
st.title("🎁 Targeted Marketing Offers")

df = load_customer_data()

# Offer distribution
st.subheader("Recommended Offers Distribution")
offer_counts = df['RecommendedOffer'].value_counts().reset_index()
offer_counts.columns = ['Offer', 'CustomerCount']
fig = px.bar(offer_counts, x='CustomerCount', y='Offer', orientation='h',
             title='Number of Customers per Offer')
st.plotly_chart(fig, use_container_width=True)

# Filter by offer
selected_offer = st.selectbox("Select an offer to see customers", df['RecommendedOffer'].unique())
offer_customers = df[df['RecommendedOffer'] == selected_offer]
st.dataframe(offer_customers[['CustomerID','Segment','ChurnProb','HistoricalLTV','RecommendedOffer']])

# Campaign cost summary (if you have that function)
st.subheader("Campaign Summary")
try:
    from src.business.offer_recommender import create_marketing_campaign_summary
    summary = create_marketing_campaign_summary(df)
    st.dataframe(summary)
except:
    st.info("Campaign summary function not available; showing basic counts.")