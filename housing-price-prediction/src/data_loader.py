import pandas as pd
import os

def load_housig_data(file_path : str) ->pd.DataFrame:
    """Loads the housing data set from a given path"""
    df = pd.read_csv(file_path)
    return df