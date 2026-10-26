import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

mlflow.set_experiment("iris-classification-baseline")

data = pd.read_csv("data/processed/iris_features.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest_shallow": RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42
    ),
    "random_forest_deep": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
}

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="macro"
        )
        recall = recall_score(
            y_test, predictions, average="macro"
        )
        f1 = f1_score(
            y_test, predictions, average="macro"
        )

        mlflow.log_param("model_name", model_name)

        if model_name == "logistic_regression":
            mlflow.log_param("max_iter", 200)

        elif model_name == "random_forest_shallow":
            mlflow.log_param("n_estimators", 50)
            mlflow.log_param("max_depth", 3)

        elif model_name == "random_forest_deep":
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", "None")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)

        cm = confusion_matrix(y_test, predictions)

        cm_file = f"confusion_matrix_{model_name}.txt"

        with open(cm_file, "w") as f:
            f.write(str(cm))

        mlflow.log_artifact(cm_file)

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print(f"\nModel: {model_name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(
            f"Run ID:    {mlflow.active_run().info.run_id}"
        )

print("\nAll models trained and logged to MLflow.")
