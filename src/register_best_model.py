import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "iris-classification-baseline"
MODEL_NAME = "iris-classifier-prod"

client = MlflowClient()

# Find the experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError(
        f"Experiment '{EXPERIMENT_NAME}' not found."
    )

# Find the run with the highest F1 score
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_macro DESC"],
    max_results=1
)

if not runs:
    raise RuntimeError("No MLflow runs found.")

best_run = runs[0]

run_id = best_run.info.run_id
f1_score = best_run.data.metrics["f1_macro"]

model_uri = f"runs:/{run_id}/model"

print(f"Best Run ID: {run_id}")
print(f"Best F1 Score: {f1_score:.4f}")
print(f"Model URI: {model_uri}")

# Register the model
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print(
    f"Registered {MODEL_NAME} "
    f"version {registered_model.version}"
)

# Move model to Staging
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=registered_model.version,
    stage="Staging"
)

print(f"{MODEL_NAME} is now in Staging.")
