import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_historical_ltv(df_transactions,
                             customer_id='CustomerID',
                             monetary_col='TotalPrice',
                             period_days=None):
    """
    Calculate historical LTV (total spend) for each customer.
    If period_days is provided, only consider transactions in last N days.
    """
    if period_days:
        end_date = df_transactions['InvoiceDate'].max()
        start_date = end_date - timedelta(days=period_days)
        df_period = df_transactions[df_transactions['InvoiceDate'] >= start_date]
        logger.info(f"Calculating LTV over last {period_days} days ({start_date} to {end_date})")
    else:
        df_period = df_transactions
        logger.info("Calculating historical LTV over entire dataset")

    ltv_hist = df_period.groupby(customer_id)[monetary_col].sum().rename('HistoricalLTV')
    return ltv_hist


def prepare_ltv_features(customer_features,
                         churn_labels=None,
                         include_churn=True):
    """
    Prepare features for predictive LTV model.
    Features include RFM, tenure, average order value, etc.
    """
    # Start with all numeric features from customer_features (excluding LTV if present)
    feature_cols = ['Recency', 'Frequency', 'Monetary', 'TenureDays', 'AvgOrderValue']
    X = customer_features[feature_cols].copy()

    if include_churn and churn_labels is not None:
        # Align churn labels (assuming same index)
        X['ChurnLabel'] = churn_labels

    return X


def train_predictive_ltv_model(X, y_ltv, model_type='random_forest', test_size=0.2):
    """
    Train a regression model to predict future LTV (or total LTV).
    y_ltv is the target variable (e.g., LTV over next 12 months).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_ltv, test_size=test_size, random_state=42
    )

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("model_type must be 'linear' or 'random_forest'")

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Model: {model_type}")
    logger.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

    return model, X_train, X_test, y_train, y_test


def save_ltv_model(model, filepath='models/ltv_model.pkl'):
    joblib.dump(model, filepath)
    logger.info(f"LTV model saved to {filepath}")


def load_ltv_model(filepath='models/ltv_model.pkl'):
    model = joblib.load(filepath)
    logger.info(f"LTV model loaded from {filepath}")
    return model