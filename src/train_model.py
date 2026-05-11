from sklearn.ensemble import RandomForestRegressor
from src.predict import evaluate_model
from sklearn.model_selection import GridSearchCV


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


def hyperparameter_tuning(X_train, y_train):
    """
    perform hyperparameter tuning using GridSearchCV.
    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
    Returns: The best model found by GridSearchCV.
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)
    return best_model
