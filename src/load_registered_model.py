import mlflow
import pandas as pd

model_uri = "models:/iris-classifier-prod/Staging"

print("Loading registered model...")
model = mlflow.pyfunc.load_model(model_uri)

print("Model loaded successfully!")

data = pd.read_csv("data/processed/iris_features.csv")

X = data.drop("target", axis=1)
y = data["target"]

predictions = model.predict(X.head(5))

print("Sample predictions:")
print(predictions)
