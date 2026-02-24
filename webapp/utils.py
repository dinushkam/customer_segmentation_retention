import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path
from datetime import timedelta

# ======================
# ORIGINAL FUNCTIONS (keep these)
# ======================

@st.cache_data(ttl=3600)
def load_customer_data():
    """Load the final customer dataset with offers."""
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'customers_with_offers.csv'
    if not data_path.exists():
        st.error(f"❌ Data file not found at {data_path}. Please run the pipeline first.")
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    df['CustomerID'] = df['CustomerID'].astype(str)
    return df

@st.cache_resource
def load_churn_model():
    model_path = Path(__file__).parent.parent / 'models' / 'churn_model_rf.pkl'
    if not model_path.exists():
        st.error("❌ Churn model not found.")
        return None
    return joblib.load(model_path)

@st.cache_resource
def load_ltv_model():
    model_path = Path(__file__).parent.parent / 'models' / 'ltv_model_rf.pkl'
    if not model_path.exists():
        st.warning("⚠️ LTV model not found. Predictive LTV will be unavailable.")
        return None
    return joblib.load(model_path)

def get_feature_columns():
    return ['Recency', 'Frequency', 'Monetary', 'TenureDays', 'AvgOrderValue']

def metric_card(title, value, delta=None, delta_color="normal", help_text=""):
    """Generate HTML for a metric card."""
    delta_class = "positive" if delta and delta.startswith('+') else "negative" if delta and delta.startswith('-') else ""
    delta_html = f'<p class="delta {delta_class}">{delta}</p>' if delta else ''
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <p class="value">{value}</p>
        {delta_html}
        {help_text}
    </div>
    """

# ======================
# NEW FUNCTIONS FOR UPLOAD PROCESSING
# ======================

def clean_transaction_data(df):
    """Basic cleaning: remove missing CustomerID, negative quantities/prices, create TotalPrice."""
    required = ['CustomerID', 'InvoiceDate']
    if not all(col in df.columns for col in required):
        st.error(f"Uploaded file must contain columns: {required}")
        return None

    df_clean = df.copy()
    df_clean.dropna(subset=['CustomerID'], inplace=True)
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(str)

    # If TotalPrice not present, create from Quantity and UnitPrice
    if 'TotalPrice' not in df_clean.columns:
        if 'Quantity' in df_clean.columns and 'UnitPrice' in df_clean.columns:
            df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
        else:
            st.error("Need either 'TotalPrice' or both 'Quantity' and 'UnitPrice'.")
            return None

    # Remove negative or zero quantities/prices (optional)
    df_clean = df_clean[df_clean['TotalPrice'] > 0]
    return df_clean

def compute_rfm(df, reference_date=None):
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'CustomerID': 'count',
        'TotalPrice': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'CustomerID': 'Frequency',
        'TotalPrice': 'Monetary'
    })

    # Score (1-5) using quantiles
    try:
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    except:
        rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
    try:
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    except:
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    try:
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
    except:
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])

    # 🚨 Drop any customers with missing scores
    rfm.dropna(subset=['R_Score', 'F_Score', 'M_Score'], inplace=True)

    # ✅ Convert to integer
    rfm['R_Score'] = rfm['R_Score'].astype(int)
    rfm['F_Score'] = rfm['F_Score'].astype(int)
    rfm['M_Score'] = rfm['M_Score'].astype(int)

    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # Segment mapping (simplified)
    def assign_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']  # now ints
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal'
        elif r <= 2 and (f >= 4 or m >= 4):
            return 'At Risk'
        elif r >= 4 and f <= 2 and m <= 2:
            return 'New'
        else:
            return 'Others'
    rfm['Segment'] = rfm.apply(assign_segment, axis=1)

    return rfm
    # Segment mapping (simplified)
    def assign_segment(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal'
        elif r <= 2 and (f >= 4 or m >= 4):
            return 'At Risk'
        elif r >= 4 and f <= 2 and m <= 2:
            return 'New'
        else:
            return 'Others'
    rfm['Segment'] = rfm.apply(assign_segment, axis=1)

    return rfm

def process_uploaded_data(raw_df):
    """Main function to process uploaded transaction data into customer features."""
    cleaned = clean_transaction_data(raw_df)
    if cleaned is None:
        return None

    # Compute RFM
    rfm = compute_rfm(cleaned)

    # Add historical LTV (same as Monetary)
    rfm['HistoricalLTV'] = rfm['Monetary']

    # Add tenure (days since first purchase)
    first_purchase = cleaned.groupby('CustomerID')['InvoiceDate'].min().rename('FirstPurchase')
    rfm = rfm.join(first_purchase)
    ref_date = cleaned['InvoiceDate'].max()
    rfm['TenureDays'] = (ref_date - rfm['FirstPurchase']).dt.days
    rfm.drop('FirstPurchase', axis=1, inplace=True)

    # Average order value
    avg_order = cleaned.groupby('CustomerID')['TotalPrice'].mean().rename('AvgOrderValue')
    rfm = rfm.join(avg_order)

    # Simple churn probability: based on recency (e.g., logistic function)
    # Here we use a simple threshold: if recency > 90 days, prob = 0.8, else scaled
    max_recency = rfm['Recency'].max()
    rfm['ChurnProb'] = rfm['Recency'].apply(lambda x: min(1.0, x / 180))  # linear increase

    # Predicted LTV (6 months): simple heuristic – historical LTV scaled by recency factor
    rfm['PredictedLTV_Next6Months'] = rfm['HistoricalLTV'] * (1 - rfm['ChurnProb'] * 0.5)

    # Reset index to make CustomerID a column
    rfm.reset_index(inplace=True)
    rfm.rename(columns={'index': 'CustomerID'}, inplace=True)

    return rfm