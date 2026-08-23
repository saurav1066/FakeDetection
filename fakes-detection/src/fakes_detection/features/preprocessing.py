from fakes_detection.data.load import load_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


from pathlib import Path
#Covering multiple date formatsimport pandas as pd

#covering multiple date formats
def parse_dates(date):
    for fmt in ('%Y-%d-%m', '%m/%d/%y'):  # adjusting formats as found the data inspection
        try:
            return pd.to_datetime(date, format=fmt)
        except ValueError:
            continue
    return pd.NaT 




def build_preprocessor(dataset_path) -> pd.DataFrame:

    # loading the dataset from the specified path
    df = load_dataset(dataset_path)

    #parsing date in the dataset to a standard format

    #Converting Purchase D  data['PurchDate'] = data['PurchDate'].apply(parse_dates)
    df['PurchDate'] = pd.to_datetime(df['PurchDate'], errors='coerce')

    #Dropping unnecessary columns(Unnamed and RefId)
    df.drop(columns=['Unnamed: 0','RefId'], inplace=True)

     #Handling missing values, removing na's
    df.dropna(subset=['MarketDate'], inplace=True)
    df = df.dropna(subset=['PurchDate'])

    # Creating nre feature as there is 2 types of prices available
    df['PriceDiscrepancy'] = df['TransactionPrice'] - df['AveragePrice']

    return df

def preprocessing_pipeline(dataset_path)->tuple:
    df = build_preprocessor(dataset_path)

    X = df.drop(columns=['Fake'])
    y = df['Fake']
    # Define the preprocessing steps for numerical and categorical features
    categorical = X.select_dtypes(include=['object', 'category']).columns
    numerical = X.select_dtypes(include=['int64', 'float64']).columns

    # Create a preprocessing pipeline
    numerical_t = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
    ])

    categorical_t = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Creating the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_t, numerical),
            ('cat', categorical_t, categorical)
        ])

    return X,y,preprocessor




if __name__ == "__main__":
    dataset_path = Path("data/raw/data_fakes.csv")
    preprocessed_df = build_preprocessor(dataset_path)
    print(preprocessed_df.head())


