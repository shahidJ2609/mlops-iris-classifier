import mlflow
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "iris-hyperparameter-tuning"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.cv_f1_macro_mean DESC"]
)

print("\n=== Hyperparameter Tuning Comparison ===\n")

for run in runs:

    run_name = run.data.tags.get(
        "mlflow.runName",
        "Unknown"
    )

    cv_score = run.data.metrics.get(
        "cv_f1_macro_mean",
        0
    )

    test_accuracy = run.data.metrics.get(
        "test_accuracy",
        0
    )

    if "grid_search" in run_name:
        total_fits = 360
    elif "random_search" in run_name:
        total_fits = 150
    elif "baseline" in run_name:
        total_fits = 5
    else:
        total_fits = "N/A"

    print(f"Run Name: {run_name}")
    print(f"CV F1 Macro: {cv_score:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Total Fits: {total_fits}")
    print("-" * 40)
