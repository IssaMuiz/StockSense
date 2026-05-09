from sklearn.ensemble import RandomForestRegressor
from src.predict import evaluate_model


def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor model.
    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        Returns: The trained model."""

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    """
    Train the model and evaluate it on the validation set.
    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.
    Returns: A dictionary containing the evaluation metrics."""

    model = train_model(X_train, y_train)
    y_pred, evaluation_results = evaluate_model(model, X_val, y_val)
    return y_pred, evaluation_results
