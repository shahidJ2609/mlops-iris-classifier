import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}


def load_dataset(path):
    df = pd.read_csv(path)

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def run_grid_search(data_path):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-hyperparameter-tuning")

    X_train, X_test, y_train, y_test = load_dataset(data_path)

    with mlflow.start_run(run_name="grid_search_random_forest"):

        model = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=PARAM_GRID,
            scoring="f1_macro",
            cv=5,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        test_accuracy = best_model.score(X_test, y_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_params(grid_search.best_params_)

        mlflow.log_metric(
            "cv_f1_macro_mean",
            grid_search.best_score_
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv(
            "grid_search_all_candidates.csv",
            index=False
        )

        mlflow.log_artifact("grid_search_all_candidates.csv")

        print(f"Best CV f1_macro: {grid_search.best_score_:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Total fits: {len(results) * 5}")


if __name__ == "__main__":
    run_grid_search(
        "data/processed/iris_features.csv"
    )
