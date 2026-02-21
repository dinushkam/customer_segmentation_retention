import pandas as pd
import logging
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw Excel data."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_excel(filepath)
    logger.info(f"Data loaded with shape: {df.shape}")
    return df

def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """Save processed DataFrame to CSV."""
    logger.info(f"Saving processed data to {filepath}")
    df.to_csv(filepath, index=False)