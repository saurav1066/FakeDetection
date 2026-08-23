from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)



def evaluate_model(
    model,
    X_test,
    y_test,
) -> dict[str, float]:

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


if __name__ == "__main__":
    from fakes_detection.models.model import build_model
    from fakes_detection.features.preprocessing import preprocessing_pipeline
    from fakes_detection.training.train import train_model
    from fakes_detection.inference.predict import predict
    from fakes_detection.inference.predict import load_model
    import pandas as pd

    # Load the model
    model = build_model()

    # Load the test data
    data_path = "data/raw/data_fakes.csv"

    x_test, y_test= train_model(data_path, "artifacts/model.joblib")
    model = load_model("artifacts/model.joblib")
    predictions = predict(model, x_test)
    

    # Evaluate the model
    results = evaluate_model(model, x_test, y_test)
    print(results)