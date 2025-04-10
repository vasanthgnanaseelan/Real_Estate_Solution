
from src.data_loader import fetch_dataset
from src.eda_tools import explore_data
from src.preprocessing import clean_and_encode
from src.data_splitter import partition_data
from src.model_trainer import train_random_forest
from src.model_eval import assess_model
import logging

def main():
    # Local data file path
    file_path = r"E:\BISI\Machine learning\Project\Real_Estate_Solution\data\final.csv"
    
    # Fetch dataset
    df = fetch_dataset(file_path)
    if df.empty:
        logging.error("Dataset could not be loaded. Exiting.")
        return
    
    # Perform basic EDA
    explore_data(df)

    # Preprocess the data (handle missing values, encoding)
    df = clean_and_encode(df)
    
    # Split the data into training and testing sets
    try:
        X_train, X_test, y_train, y_test = partition_data(df, "price")
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        return
    
    # Train the model
    model = train_random_forest(X_train, y_train)
    if model is None:
        logging.error("Model training failed. Exiting.")
        return
    
    # Evaluate the model
    try:
        mae, mse = assess_model(model, X_test, y_test)
        logging.info(f"Model Evaluation:\nMAE: {mae}\nMSE: {mse}")
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return

if __name__ == "__main__":
    main()
