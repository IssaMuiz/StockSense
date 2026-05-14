from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pipeline.features import (
    num_features,
    cat_features,
)
from pipeline.build_pipeline import num_pipeline, cat_pipeline

# Build the complete preprocessing pipeline
pipeline_preprocessing = Pipeline(
    steps=[
        (
            "column_transformer",
            ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        num_features,
                    ),
                    ("cat_pipeline", cat_pipeline, cat_features),
                ],
                remainder="drop",
                n_jobs=-1,
            ),
        ),
    ],
)
