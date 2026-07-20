import pandas as pd
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """Loads the housing dataset from a given path."""
    return pd.read_csv(file_path)


def num_cat_cols_lst(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Returns lists of numerical and categorical column names."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    return num_cols, cat_cols


def general_desc(df: pd.DataFrame) -> None:
    """Prints a general summary of the dataset."""
    n_rows = len(df)
    n_cols = len(df.columns)
    
    num_cols, cat_cols = num_cat_cols_lst(df)
    
    n_missing = df.isna().sum().sum()

    print(f"Number of records (rows): {n_rows}")
    print(f"Number of features (columns): {n_cols}")
    print(f"Number of numerical features: {len(num_cols)}")
    print(f"Number of categorical features: {len(cat_cols)}")
    print(f"Total missing values: {n_missing}")


def data_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a general summary of data statistics."""
    return df.describe().reset_index()


def feat_eng(df: pd.DataFrame) -> pd.DataFrame:
    """Applies feature engineering to create new variables."""
    
    return df.assign(
        bathrooms_per_bedroom=np.where(
            df['bedrooms'] > 0,
            df['bathrooms'] / df['bedrooms'],
            df['bathrooms']
        )
    )


def corr_mat(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and returns the correlation matrix for numerical features."""
    num_cols, _ = num_cat_cols_lst(df)
    return df[num_cols].corr()