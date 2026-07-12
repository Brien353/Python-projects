import numpy as np
import pandas as pd
import umap
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from config.clust_config import num_cl_feat, cat_cl_feat


def num_pipe() -> Pipeline:
    """Handles numerical missing values and scales features."""
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

def cat_pipe() -> Pipeline:
    """Handles categorical missing values and encodes them to dense vectors."""
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    )

def prep_pipe() -> ColumnTransformer:
    """Combines the numerical and categorical parallel processing branches."""
    return ColumnTransformer(
        transformers=[
            ('num', num_pipe(), num_cl_feat),
            ('cat', cat_pipe(), cat_cl_feat)
        ]
    )

def projection_pipe() -> Pipeline:
    """Main pipeline that preprocesses data and runs UMAP dimensionality reduction."""
    return Pipeline(
        steps=[
            ('prep', prep_pipe()),
            ('umap', umap.UMAP(
                n_components=2,
                n_neighbors=20, 
                min_dist=0.0,
                metric='euclidean',
                random_state=42
            ))
        ]
    )








