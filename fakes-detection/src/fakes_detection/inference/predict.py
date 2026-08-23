import joblib
import pandas as pd
from fakes_detection.features.preprocessing import preprocessing_pipeline


def load_model(path: str):
    return joblib.load(path)


def predict(model, data: pd.DataFrame):
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    return {
        "predictions": predictions,
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    model_path = "artifacts/model.joblib"
    data_path = "data/raw/data_fakes.csv"

    model = load_model(model_path)
    data = pd.read_csv(data_path)
    X, y, preprocessing = preprocessing_pipeline(data_path)

    results = predict(model, X)
    print(results)
