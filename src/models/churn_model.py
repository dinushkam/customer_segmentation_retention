import pandas as pd
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from src.features.build_features import build_all_features
import logging

logger = logging.getLogger(__name__)


def prepare_split_data(df: pd.DataFrame, churn_window_days: int = 90):
    """
    Prevents Data Leakage by using a Time-Slicing approach.
    Data is split into:
    1. Feature Window: Everything before (MaxDate - churn_window_days)
    2. Label Window: The final churn_window_days to determine if they stayed.
    """
    max_date = df['InvoiceDate'].max()
    cutoff_date = max_date - timedelta(days=churn_window_days)

    # Data used to create features (History)
    feature_data = df[df['InvoiceDate'] < cutoff_date]
    # Data used to check for 'future' activity (Labels)
    label_data = df[df['InvoiceDate'] >= cutoff_date]

    # 1. Build features ONLY from history
    X = build_all_features(feature_data, reference_date=cutoff_date)

    # 2. Define Labels: 1 if customer did NOT appear in label_data, else 0
    active_customers = label_data['CustomerID'].unique()
    y = X.index.map(lambda x: 0 if x in active_customers else 1)

    return X, y


def train_churn_model_robust(X, y):
    """Trains model using TimeSeriesSplit to ensure temporal validity."""
    # Use selected numeric features
    feature_cols = ['Recency', 'Frequency', 'Monetary', 'TenureDays', 'AvgOrderValue', 'StdDaysBetweenOrders']
    X_train = X[feature_cols].fillna(0)

    # TimeSeriesSplit is better for transaction data than random split
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

    for train_index, test_index in tscv.split(X_train):
        X_t, X_v = X_train.iloc[train_index], X_train.iloc[test_index]
        y_t, y_v = y[train_index], y[test_index]
        model.fit(X_t, y_t)

    return model