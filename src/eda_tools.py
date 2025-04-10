import logging

def explore_data(df):
    """
    Outputs summary of dataset: dimensions, types, and missing values.
    """
    try:
        logging.info("Performing EDA summary")
        print(f"Shape: {df.shape}")
        print("\nData Types:\n", df.dtypes)
        print("\nMissing Values:\n", df.isnull().sum())
        print("\nDescriptive Statistics:\n", df.describe())
    except Exception as err:
        logging.error(f"EDA error: {err}")
        print("An error occurred during EDA. Please check the logs for more details.")      