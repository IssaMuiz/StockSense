from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on the validation set.
    Args:
        model: The trained model.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validatio target variable.
        Returns: A dictionary containing the evaluation metrics."""

    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    evaluation_results = {
        "Mean Squared Error": mse,
        "Mean Absolute Error": mae,
        "R2 Score": r2,
    }
    return y_pred, evaluation_results
