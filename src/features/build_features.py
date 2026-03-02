import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_rfm(df: pd.DataFrame,
                  customer_id: str = 'CustomerID',
                  date_col: str = 'InvoiceDate',
                  monetary_col: str = 'TotalPrice',
                  reference_date=None) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary (RFM) scores with robust ranking."""
    if reference_date is None:
        reference_date = df[date_col].max() + timedelta(days=1)

    rfm = df.groupby(customer_id).agg({
        date_col: lambda x: (reference_date - x.max()).days,
        customer_id: 'count',
        monetary_col: 'sum'
    }).rename(columns={
        date_col: 'Recency',
        customer_id: 'Frequency',
        monetary_col: 'Monetary'
    })

    # Use rank-based quantiles to handle tied values in Frequency/Recency
    for col, labels in zip(['Recency', 'Frequency', 'Monetary'], [[5, 4, 3, 2, 1], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]):
        rfm[f'{col[0]}_Score'] = pd.qcut(rfm[col].rank(method='first'), 5, labels=labels).astype(int)

    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    def assign_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 4 and f >= 4 and m >= 4: return 'Champions'
        if r >= 3 and f >= 3 and m >= 3: return 'Loyal'
        if r <= 2 and (f >= 4 or m >= 4): return 'At Risk'
        if r >= 4 and f <= 2: return 'New'
        return 'Others'

    rfm['Segment'] = rfm.apply(assign_segment, axis=1)
    return rfm


def add_purchase_frequency_features(df: pd.DataFrame, customer_id: str = 'CustomerID',
                                    date_col: str = 'InvoiceDate') -> pd.DataFrame:
    """Calculates the regularity of purchases (Standard Deviation of Days between orders)."""
    df_sorted = df.sort_values([customer_id, date_col])
    # Calculate days between consecutive purchases per customer
    df_sorted['Diff'] = df_sorted.groupby(customer_id)[date_col].diff().dt.days

    freq_stats = df_sorted.groupby(customer_id)['Diff'].agg(['mean', 'std']).fillna(0)
    freq_stats.columns = ['AvgDaysBetweenOrders', 'StdDaysBetweenOrders']
    return freq_stats


def build_all_features(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Master function to run all feature engineering."""
    rfm = calculate_rfm(df, reference_date=reference_date)
    # Tenure and Average Order Value
    tenure = df.groupby('CustomerID')['InvoiceDate'].agg(['min', 'max'])
    tenure['TenureDays'] = (df['InvoiceDate'].max() - tenure['min']).dt.days
    avg_order = df.groupby('CustomerID')['TotalPrice'].mean().rename('AvgOrderValue')

    # New Frequency Regularity Features
    freq_regularity = add_purchase_frequency_features(df)

    features = rfm.join(tenure[['TenureDays']]).join(avg_order).join(freq_regularity)
    return features