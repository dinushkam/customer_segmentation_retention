import pandas as pd
import numpy as np
from datetime import timedelta
import logging
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def define_churn_labels(df_transactions,
                        customer_features,
                        customer_id='CustomerID',
                        date_col='InvoiceDate',
                        churn_window_days=90,
                        reference_date=None):
    """
    Define churn labels for each customer based on whether they made a purchase
    in the last 'churn_window_days' relative to reference_date.

    Parameters:
    -----------
    df_transactions : pd.DataFrame
        Transaction data with at least customer_id and date_col.
    customer_features : pd.DataFrame
        DataFrame with customer_id as index (from feature engineering).
    churn_window_days : int
        Number of days without purchase to be considered churned.
    reference_date : datetime, optional
        The date to use as 'today'. If None, uses max date in transactions.

    Returns:
    --------
    pd.Series with churn labels (1 = churned, 0 = active) aligned with customer_features index.
    """
    if reference_date is None:
        reference_date = df_transactions[date_col].max()

    # Get last purchase date per customer
    last_purchase = df_transactions.groupby(customer_id)[date_col].max()

    # Compute days since last purchase
    days_since_last = (reference_date - last_purchase).dt.days

    # Align with customer_features index
    days_since_last = days_since_last.reindex(customer_features.index,
                                              fill_value=churn_window_days + 1)  # if missing, assume churned

    # Churn = 1 if days_since_last > churn_window_days else 0
    churn = (days_since_last > churn_window_days).astype(int)

    logger.info(f"Churn label distribution: {churn.value_counts().to_dict()}")
    return churn


def prepare_features_and_labels(customer_features, churn_labels, feature_cols=None):
    """
    Prepare feature matrix X and target vector y.
    """
    if feature_cols is None:
        # Select numeric columns (excluding scores that might be categorical)
        feature_cols = ['Recency', 'Frequency', 'Monetary', 'TenureDays', 'AvgOrderValue']

    X = customer_features[feature_cols].copy()
    y = churn_labels.copy()

    # Handle missing values (should be none if feature engineering is clean)
    if X.isnull().any().any():
        logger.warning("Missing values found in features. Filling with median.")
        X.fillna(X.median(), inplace=True)

    return X, y


def train_churn_model(X, y, model_type='random_forest', tune_hyperparams=False):
    """
    Train a churn prediction model with optional hyperparameter tuning.

    model_type: 'random_forest' or 'xgboost'
    """
    logger.info(f"Training {model_type} model...")

    # Train/test split (temporal split recommended for time-series, but we'll use random for simplicity)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            logger.info(f"Best parameters: {grid.best_params_}")
        else:
            model.fit(X_train, y_train)
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(random_state=42, scale_pos_weight=(len(y) - sum(y)) / sum(y))  # handle imbalance
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1]
            }
            grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            logger.info(f"Best parameters: {grid.best_params_}")
        else:
            model.fit(X_train, y_train)
    else:
        raise ValueError("model_type must be 'random_forest' or 'xgboost'")

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    # Feature importance
    if model_type == 'random_forest':
        importances = model.feature_importances_
    else:  # xgboost
        importances = model.feature_importances_

    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    logger.info(f"Top 5 features:\n{feat_imp.head(5)}")

    return model, X_train, X_test, y_train, y_test


def save_model(model, filepath='models/churn_model.pkl'):
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath='models/churn_model.pkl'):
    """Load trained model from disk."""
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model