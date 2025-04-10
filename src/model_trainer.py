from sklearn.ensemble import RandomForestRegressor
import logging

def train_random_forest(X, y):
    """
    Trains a RandomForestRegressor model and returns it.
    """
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        logging.info("Random Forest model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Model training error: {e}")
        return None
