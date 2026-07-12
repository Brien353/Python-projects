import pandas as pd
import os
import numpy as np

#Load data

def load_housig_data(file_path : str) ->pd.DataFrame:
    """Loads the housing data set from a given path"""
    df = pd.read_csv(file_path)
    return df

# Select numerical and categorical columns

def num_cat_cols_lst(df: pd.DataFrame)-> tuple[list,list]:
    """ Return the list of numerical and categorical columns of a data frame"""
    num_cols = df.select_dtypes(include = np.number).columns.to_list()
    cat_cols = df.select_dtypes(include = 'object').columns.to_list()
    return num_cols,cat_cols

#split data

def general_desc(df:pd.DataFrame)-> tuple[str,...]:
    """ Gives a general resume about data"""

    # Count database registers
    n_registers = len(df)
    print(f'The number of registers of the data base is {n_registers}')
    # Count the number of features of the database
    n_features = len(df.columns)
    print(f'The number of features of the data base is {n_features}')
    # Count the number of categorical features of the database
    _,n_categorical_features = num_cat_cols_lst(df)
    print(f'The number of categorical features of the data base is {len(n_categorical_features)}')
    #Count the number of numerical features of the database
    n_numerical_features, _  = num_cat_cols_lst(df)
    print(f'The number of numerical features of the data base is {len(n_numerical_features)}')
    #Count the number of null values of the database
    n_null_values = df.isnull().sum().sum()
    print(f'The number of null values of the data base is {n_null_values}')
    # Count the number of missing values of the database
    n_missing_values = df.isna().sum().sum()
    print(f'The number of missing values of the data base is {n_missing_values}')

def data_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create a general resume about data statistics"""
    resume = pd.DataFrame(df.describe()).reset_index()
    return resume


def feat_eng(df:pd.DataFrame)-> pd.DataFrame:
    df['bathrooms_per_bedroom'] = np.where(
    df['bedrooms'] > 0,
    df['bathrooms'] / df['bedrooms'],
    df['bathrooms'])
    return df


def corr_mat(df: pd.DataFrame)->pd.DataFrame:
    """Calculate the correlation matrix and return it"""
    num_cols , _ =num_cat_cols_lst(df)
    dg = df.copy()
    dg = dg[num_cols]
    corr_mat = pd.DataFrame(dg.corr())
    return corr_mat


