from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X_val, y_val):
    """Evaluate the model
    Args:
        model: The trained model.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.
        Returns: A dictionary containing the evaluation metrics."""

    y_pred = model.predict(X_val)

    metrics = {
        "Mean Squared Error": mean_squared_error(y_val, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_val, y_pred),
        "R2 Score": r2_score(y_val, y_pred),
    }

    return metrics
