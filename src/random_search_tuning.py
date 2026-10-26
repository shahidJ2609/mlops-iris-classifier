import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


PARAM_DIST = {
    "n_estimators": randint(50, 300),
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_split": randint(2, 15),
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


def run_random_search(data_path):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-hyperparameter-tuning")

    X_train, X_test, y_train, y_test = load_dataset(data_path)

    with mlflow.start_run(run_name="random_search_random_forest"):

        model = RandomForestClassifier(random_state=42)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=PARAM_DIST,
            n_iter=30,
            scoring="f1_macro",
            cv=5,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_

        test_accuracy = best_model.score(X_test, y_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_params(random_search.best_params_)

        mlflow.log_metric(
            "cv_f1_macro_mean",
            random_search.best_score_
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        results = pd.DataFrame(random_search.cv_results_)

        results.to_csv(
            "random_search_all_candidates.csv",
            index=False
        )

        mlflow.log_artifact(
            "random_search_all_candidates.csv"
        )

        print(
            f"Best CV f1_macro: "
            f"{random_search.best_score_:.4f}"
        )

        print(
            f"Test accuracy: "
            f"{test_accuracy:.4f}"
        )

        print(
            f"Best parameters: "
            f"{random_search.best_params_}"
        )

        print(
            f"Total fits: "
            f"{len(results) * 5}"
        )


if __name__ == "__main__":
    run_random_search(
        "data/processed/iris_features.csv"
    )
