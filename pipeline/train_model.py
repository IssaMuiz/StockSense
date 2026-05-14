from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from pipeline.transformers import pipeline_preprocessing


def run_pipeline(X_train, y_train):
    """
    Build and train a machine learning pipeline with preprocessing and RandomForestRegressor.
    Parameters:
    X_train (array-like): Training feature data.
    y_train (array-like): Training target labels.
    """

    # Create a pipeline with preprocessing and model
    pipeline = make_pipeline(
        pipeline_preprocessing, RandomForestRegressor(random_state=42)
    )

    param_grid = {
        "randomforestregressor__n_estimators": [50, 100, 200],
        "randomforestregressor__max_depth": [5, 10, None],
        "randomforestregressor__min_samples_split": [2, 5],
        "randomforestregressor__min_samples_leaf": [1, 2],
    }

    rf_grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
    )

    rf_grid.fit(X_train, y_train)

    best_estimator = rf_grid.best_estimator_

    return best_estimator  # Return the trained pipeline
