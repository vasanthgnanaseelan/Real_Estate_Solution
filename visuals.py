from src.data_loader import fetch_dataset
from src.plot_tools import kde_distribution, custom_heatmap
import logging

def main():
    # Use local file path for development
    file_path = r"E:\BISI\Machine learning\Project\Real_Estate_Solution\data\final.csv"
    
    # Load dataset
    df = fetch_dataset(file_path)
    if df.empty:
        logging.error("Dataset could not be loaded.")
        return

    # Plot KDE distribution of 'price'
    kde_distribution(df, "price")

    # Optional: plot another feature (change column name if needed)
    kde_distribution(df, "sqft")

    # Plot correlation heatmap
    custom_heatmap(df)

if __name__ == "__main__":
    main()
