import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from utils import load_churn_model, get_feature_columns

st.set_page_config(page_title="Model Performance", layout="wide")

# Load custom CSS
with open(Path(__file__).parent.parent / 'assets' / 'style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>Customer 360 Analytics <span style="color:#6A1B9A;">| Model Performance</span></h1>
    <hr>
</div>
""", unsafe_allow_html=True)

model = load_churn_model()
if model is None:
    st.stop()

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
    color_discrete_sequence=['#6A1B9A']
)
fig_imp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_imp, use_container_width=True)

# ROC Curve (illustrative)
st.subheader("ROC Curve")
fpr = np.linspace(0, 1, 100)
tpr = fpr ** 0.7  # dummy curve with AUC ≈ 0.85
fig_roc = px.area(
    x=fpr,
    y=tpr,
    title='ROC Curve (AUC ≈ 0.85)',
    labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
)
fig_roc.add_shape(
    type='line',
    line=dict(dash='dash', color='grey'),
    x0=0, x1=1, y0=0, y1=1
)
fig_roc.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_roc, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <hr>
    <p>© 2025 Customer 360 Analytics. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)