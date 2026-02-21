import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_customer_data, load_churn_model, get_feature_columns

st.set_page_config(page_title="Churn Analysis", layout="wide")
st.title("⚠️ Churn Prediction Analysis")

df = load_customer_data()
model = load_churn_model()
features = get_feature_columns()

# Churn probability distribution
st.subheader("Churn Probability Distribution")
fig = px.histogram(df, x='ChurnProb', nbins=50, title='Churn Probability Distribution')
st.plotly_chart(fig, use_container_width=True)

# High-risk customers
st.subheader("High-Risk Customers (Churn Prob > 0.7)")
high_risk = df[df['ChurnProb'] > 0.7].sort_values('ChurnProb', ascending=False)
st.dataframe(high_risk[['CustomerID','Segment','Recency','Frequency','ChurnProb']])

# Feature importance
st.subheader("Feature Importance (Random Forest)")
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
fig_imp = px.bar(feat_imp, x=feat_imp.values, y=feat_imp.index, orientation='h',
                 title='Feature Importance for Churn')
st.plotly_chart(fig_imp, use_container_width=True)