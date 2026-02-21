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
    """
    Compute Recency, Frequency, Monetary (RFM) scores and segments.

    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data with at least customer_id, date_col, monetary_col.
    customer_id : str
        Column name for customer identifier.
    date_col : str
        Column name for transaction datetime.
    monetary_col : str
        Column name for transaction value.
    reference_date : datetime, optional
        Date to use as 'now' for recency calculation. If None, uses max date in df + 1 day.

    Returns:
    --------
    pd.DataFrame with index = customer_id, columns = Recency, Frequency, Monetary,
                  R_Score, F_Score, M_Score, RFM_Score, Segment.
    """
    logger.info("Starting RFM calculation...")

    if reference_date is None:
        reference_date = df[date_col].max() + timedelta(days=1)
        logger.info(f"Reference date set to {reference_date}")

    # Group by customer
    rfm = df.groupby(customer_id).agg({
        date_col: lambda x: (reference_date - x.max()).days,  # Recency in days
        customer_id: 'count',  # Frequency (number of transactions)
        monetary_col: 'sum'  # Monetary (total spend)
    }).rename(columns={
        date_col: 'Recency',
        customer_id: 'Frequency',
        monetary_col: 'Monetary'
    })

    # Handle potential edge cases (e.g., zero recency? it's fine)
    # Remove any customers with zero recency? No, keep them.

    # Create quartile-based scores (1-5, 5 is best)
    try:
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])  # lower recency = higher score
    except ValueError:
        # If not enough unique values for quantiles, fallback to rank-based
        rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])

    try:
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    except ValueError:
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

    try:
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    except ValueError:
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

    # Combine scores as string
    rfm['RFM_Score'] = (rfm['R_Score'].astype(str) +
                        rfm['F_Score'].astype(str) +
                        rfm['M_Score'].astype(str))

    # Segment assignment (customizable)
    def assign_segment(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        # Champions: bought recently, most often, and spent the most
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        # Loyal customers: average frequency and monetary, high recency
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal'
        # At Risk: haven't purchased recently but used to buy frequently/spend a lot
        elif r <= 2 and (f >= 4 or m >= 4):
            return 'At Risk'
        # New customers: recent purchase but low frequency and monetary
        elif r >= 4 and f <= 2 and m <= 2:
            return 'New'
        # Others (can be further refined)
        else:
            return 'Others'

    rfm['Segment'] = rfm.apply(assign_segment, axis=1)

    logger.info(f"RFM calculation complete. Segments: {rfm['Segment'].value_counts().to_dict()}")
    return rfm


def add_tenure_features(df: pd.DataFrame,
                        customer_id: str = 'CustomerID',
                        date_col: str = 'InvoiceDate',
                        reference_date=None) -> pd.DataFrame:
    """
    Add customer tenure features: first purchase date, last purchase date, days since first purchase.
    Returns a DataFrame with customer_id as index and new columns.
    """
    if reference_date is None:
        reference_date = df[date_col].max()

    tenure = df.groupby(customer_id)[date_col].agg(['min', 'max']).rename(columns={
        'min': 'FirstPurchase',
        'max': 'LastPurchase'
    })
    tenure['TenureDays'] = (reference_date - tenure['FirstPurchase']).dt.days
    tenure['RecencyDays'] = (reference_date - tenure['LastPurchase']).dt.days
    return tenure


def add_avg_order_value(df: pd.DataFrame,
                        customer_id: str = 'CustomerID',
                        monetary_col: str = 'TotalPrice') -> pd.Series:
    """Calculate average order value per customer."""
    avg_order = df.groupby(customer_id)[monetary_col].mean().rename('AvgOrderValue')
    return avg_order


def add_purchase_frequency_features(df: pd.DataFrame,
                                    customer_id: str = 'CustomerID',
                                    date_col: str = 'InvoiceDate') -> pd.DataFrame:
    """
    Add features like purchase regularity (std dev of interpurchase times),
    number of unique products purchased, etc.
    """
    # Interpurchase times (requires sorted dates per customer)
    # This is more complex; we'll implement a simplified version
    # For now, just return empty
    # We'll expand later if needed
    pass


def build_all_features(df: pd.DataFrame,
                       customer_id: str = 'CustomerID',
                       date_col: str = 'InvoiceDate',
                       monetary_col: str = 'TotalPrice') -> pd.DataFrame:
    """
    Master function to run all feature engineering steps and return a combined DataFrame.
    """
    logger.info("Building all customer features...")

    # RFM
    rfm = calculate_rfm(df, customer_id, date_col, monetary_col)

    # Tenure
    tenure = add_tenure_features(df, customer_id, date_col)

    # Average order value
    avg_order = add_avg_order_value(df, customer_id, monetary_col)

    # Combine
    features = rfm.join(tenure).join(avg_order)

    logger.info(f"Features built for {len(features)} customers.")
    return features