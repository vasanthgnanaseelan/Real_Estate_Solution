import seaborn as sns
import matplotlib.pyplot as plt
import logging

def kde_distribution(df, column):
    """
    Plots a KDE distribution for a numerical feature.
    """
    try:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(df[column], fill=True, color='skyblue')
        plt.title(f"{column} Distribution")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"KDE plot error for {column}: {e}")

def custom_heatmap(df):
    """
    Plots a correlation heatmap using seaborn.
    """
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Heatmap plot error: {e}")
