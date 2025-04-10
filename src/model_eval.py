from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

def assess_model(model, X_test, y_test):
    """
    Evaluates the model using MAE and MSE.
    """
    try:
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        logging.info("Model evaluation complete")
        return mae, mse
    except Exception as err:
        logging.error(f"Model evaluation error: {err}")
        raise
