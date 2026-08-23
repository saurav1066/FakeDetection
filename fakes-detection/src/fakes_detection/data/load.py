from pathlib import Path
import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty")

    return df


if __name__ == "__main__":
    dataset_path = Path("data/raw/data_fakes.csv")
    df = load_dataset(dataset_path)
    print(df.head())