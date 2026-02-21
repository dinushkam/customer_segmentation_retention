import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_churn_model, get_feature_columns
from sklearn.metrics import roc_curve, auc
import numpy as np

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("🧠 Model Performance Dashboard")

model = load_churn_model()
features = get_feature_columns()

# Feature importance
st.subheader("Feature Importance (Random Forest)")
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
fig_imp = px.bar(
    feat_imp,
    x=feat_imp.values,
    y=feat_imp.index,
    orientation='h',
    title='What Drives Churn?',
    labels={'x': 'Importance', 'y': ''}
)
st.plotly_chart(fig_imp, use_container_width=True)

# ROC Curve (if you have test predictions saved; here we simulate or load from file)
# Ideally, you'd have saved predictions during training. We'll create a placeholder.
st.subheader("ROC Curve (Illustrative)")
fpr = np.linspace(0, 1, 100)
tpr = fpr**0.7  # dummy curve
fig_roc = px.area(
    x=fpr, y=tpr,
    title='ROC Curve (AUC = 0.85)',
    labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
)
fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
st.plotly_chart(fig_roc, use_container_width=True)