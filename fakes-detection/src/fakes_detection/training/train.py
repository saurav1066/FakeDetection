from pathlib import Path
import pandas as pd

from fakes_detection.data.load import load_dataset
from fakes_detection.models.model import build_model
from sklearn.model_selection import train_test_split
from fakes_detection.features.preprocessing import preprocessing_pipeline
import joblib
from sklearn.pipeline import Pipeline


TARGET = 'Fake'


def train_model(
    data_path: str,
    model_path: str,
) -> tuple[pd.DataFrame, pd.Series]:

    df = load_dataset(data_path)
    #loading the dataset
    X, y, preprocessing = preprocessing_pipeline(data_path)


    model = build_model()

    clf = Pipeline(steps=[
        ('preprocessor', preprocessing),
        ('classifier', model.named_steps['classifier'])
    ])
    

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf.fit(X_train, y_train)

    joblib.dump(clf, model_path)
    return X_test, y_test



if __name__ == "__main__":
    data_path = "data/raw/data_fakes.csv"
    model_path = "artifacts/model.joblib"

    train_model(data_path, model_path)
    