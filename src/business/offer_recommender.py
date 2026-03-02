import pandas as pd
import numpy as np
import logging
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_all_customer_data(customer_features_path,
                           churn_model=None,
                           ltv_model=None,
                           transactions_df=None):
    """
    Load customer features and optionally add churn probabilities and predicted LTV.
    Returns a unified DataFrame.
    """
    customers = pd.read_csv(customer_features_path, index_col=0)
    logger.info(f"Loaded {len(customers)} customers from {customer_features_path}")

    # Add churn probabilities if model provided
    if churn_model is not None and transactions_df is not None:
        # Need to compute churn probabilities using the latest data
        # For simplicity, we'll assume we have a column 'ChurnProb' already
        # Or we can compute it here using the same features
        pass  # We'll add this in the notebook

    # Add predicted LTV if model provided
    if ltv_model is not None:
        # Compute features for prediction (same as used in training)
        # For now, assume we have a column 'PredictedLTV' already
        pass

    return customers


def generate_offer_recommendations(customers_df):
    """Logic based on risk (ChurnProb) and Value (LTV)."""

    def offer_logic(row):
        # High value/Low risk
        if row['Segment'] == 'Champions' and row['ChurnProb'] < 0.3:
            return "VIP exclusive event invitation"

        # High value/High risk (The most critical group)
        if row['HistoricalLTV'] > customers_df['HistoricalLTV'].median() and row['ChurnProb'] > 0.7:
            return "High-value win-back: 25% discount + Free Shipping"

        # Low Frequency/New
        if row['Segment'] == 'New':
            return "Welcome series: 10% off second purchase"

        # General Retention
        if row['ChurnProb'] > 0.5:
            return "Personalized 'we miss you' 15% discount"

        return "Standard monthly newsletter"

    customers = customers_df.copy()
    customers['RecommendedOffer'] = customers.apply(offer_logic, axis=1)
    return customers

def create_marketing_campaign_summary(customers_with_offers):
    """
    Create a summary table for marketing teams: number of customers per offer type,
    expected reach, and estimated cost (if we assign hypothetical costs).
    """
    summary = customers_with_offers.groupby('RecommendedOffer').agg(
        CustomerCount=('CustomerID', 'count'),
        AvgLTV=('PredictedLTV_Next6Months', 'mean'),
        TotalHistoricalLTV=('HistoricalLTV', 'sum')
    ).reset_index()

    # Hypothetical cost per offer (you can adjust)
    cost_map = {
        "VIP exclusive event invitation + early access to new products": 50,
        "Loyalty points multiplier (2x for next month)": 5,
        "Personalized 'we miss you' discount (15% off)": 8,
        "Referral bonus: give $10, get $10": 10,
        "High-value at-risk: 25% off next purchase + free shipping": 20,
        "Re-engagement email with 20% off": 6,
        "Welcome series: 10% off second purchase": 4,
        "We miss you! 15% off your next order": 7,
        "Flash sale preview (24h early access)": 2,
        "Standard monthly newsletter with personalized recommendations": 1
    }
    summary['EstimatedCostPerCustomer'] = summary['RecommendedOffer'].map(cost_map).fillna(5)
    summary['TotalCost'] = summary['CustomerCount'] * summary['EstimatedCostPerCustomer']
    summary['ExpectedRevenue'] = summary['CustomerCount'] * summary['AvgLTV'] * 0.1  # assume 10% conversion

    return summary