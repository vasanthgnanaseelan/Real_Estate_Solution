import pandas as pd
import logging

def fetch_dataset(csv_path):
    """
    Reads a CSV file and returns a DataFrame.
    """
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded data from {csv_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return pd.DataFrame()