import pandas as pd
import logging

def clean_and_encode(df):
    """
    Handles missing values and encodes categorical variables using one-hot encoding.
    """
    try:
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df = pd.get_dummies(df, drop_first=True)
        logging.info("Preprocessing complete")
        return df
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        return df
