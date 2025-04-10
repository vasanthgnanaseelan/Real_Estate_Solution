from sklearn.model_selection import train_test_split
import logging

def partition_data(df, target):
    """
    Splits the DataFrame into training and testing sets.
    """
    try:
        X = df.drop(columns=[target])
        y = df[target]
        logging.info("Splitting data into training and testing sets")
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as err:
        logging.error(f"Data split error: {err}")
        raise
