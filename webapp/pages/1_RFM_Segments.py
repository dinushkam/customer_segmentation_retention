import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
from utils import load_customer_data

st.set_page_config(page_title="RFM Segments", layout="wide")

st.title("📊 RFM Analysis")
st.markdown("Explore customer segments based on Recency, Frequency, and Monetary scores.")

df = load_customer_data()

# Sidebar filters
st.sidebar.header("RFM Filters")
selected_rfm_scores = st.sidebar.multiselect(
    "RFM Score (3-digit)",
    options=df['RFM_Score'].unique(),
    default=[]
)

filtered_df = df if not selected_rfm_scores else df[df['RFM_Score'].isin(selected_rfm_scores)]

# Distribution of RFM scores
st.subheader("RFM Score Distribution")
fig_rfm_hist = px.histogram(filtered_df, x='RFM_Score', color='Segment', title='RFM Scores by Segment')
st.plotly_chart(fig_rfm_hist, use_container_width=True)

# Segment profiles
st.subheader("Segment Profiles")
profiles = filtered_df.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(1)
st.dataframe(profiles)

# 3D scatter of R,F,M
st.subheader("3D View of RFM")
fig_3d = px.scatter_3d(
    filtered_df,
    x='Recency',
    y='Frequency',
    z='Monetary',
    color='Segment',
    hover_data=['CustomerID'],
    title='RFM Space'
)
st.plotly_chart(fig_3d, use_container_width=True)