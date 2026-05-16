import sys
from pipeline.train_model import run_pipeline
from pipeline.evaluate_model import evaluate_model
from pipeline.model_registry import save_model, save_metrics
from scripts.prepare_data import X_train, X_val, y_train, y_val, X_test, y_test

sys.path.append("..")

# model
model = run_pipeline(X_train, y_train)  # Train the model

# evaluation on the validation set
metrics = evaluate_model(
    # Evaluate the model
    model,
    X_val,
    y_val,
)

# final evaluation on the test set
test_metrics = evaluate_model(
    # Evaluate on test set
    model,
    X_test,
    y_test,
)


model_path = save_model(model, version="v1")  # Save the model
metrics_path = save_metrics(test_metrics, version="v1")  # Save the metrics

print(f"Model saved to: {model_path}")
print(f"Metrics saved to: {metrics_path}")
