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


def generate_offer_recommendations(customers_df,
                                   churn_prob_col='ChurnProb',
                                   ltv_col='PredictedLTV_Next6Months',
                                   segment_col='Segment'):
    """
    Generate targeted offer recommendations based on segment, churn risk, and LTV.
    """
    # Precompute global thresholds
    high_ltv_threshold = customers_df['HistoricalLTV'].quantile(0.9)
    median_ltv = customers_df['HistoricalLTV'].median()

    def offer_logic(row):
        segment = row[segment_col]
        churn_prob = row.get(churn_prob_col, 0)
        predicted_ltv = row.get(ltv_col, 0)  # future LTV
        historical_ltv = row['HistoricalLTV']  # past total spend

        # Champions: highest value, keep them happy
        if segment == 'Champions':
            if historical_ltv > high_ltv_threshold:
                return "VIP exclusive event invitation + early access to new products"
            else:
                return "Loyalty points multiplier (2x for next month)"

        # Loyal customers: reinforce loyalty
        elif segment == 'Loyal':
            if churn_prob > 0.5:
                return "Personalized 'we miss you' discount (15% off)"
            else:
                return "Referral bonus: give $10, get $10"

        # At Risk: win them back
        elif segment == 'At Risk':
            if historical_ltv > median_ltv:
                return "High-value at-risk: 25% off next purchase + free shipping"
            else:
                return "Re-engagement email with 20% off"

        # New customers: encourage repeat purchase
        elif segment == 'New':
            return "Welcome series: 10% off second purchase"

        # Others (bulk of customers)
        else:
            if churn_prob > 0.7:
                return "We miss you! 15% off your next order"
            elif historical_ltv > median_ltv:
                return "Flash sale preview (24h early access)"
            else:
                return "Standard monthly newsletter with personalized recommendations"

    customers = customers_df.copy()
    customers['RecommendedOffer'] = customers.apply(offer_logic, axis=1)

    # Count offers for reporting
    offer_counts = customers['RecommendedOffer'].value_counts()
    logger.info(f"Offer distribution:\n{offer_counts}")

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