import umap
from config.general_settings import CAT_CL_FEAT, NUM_CL_FEAT
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def num_pipe() -> Pipeline:
    """
    Builds a pipeline for numerical features.
    Handles missing values via median imputation and scales features.
    """
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]
    )

def cat_pipe() -> Pipeline:
    """
    Builds a pipeline for categorical features.
    Handles missing values via mode imputation and applies one-hot encoding.
    """
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    )


def prep_pipe(num_features: list[str] = NUM_CL_FEAT, 
              cat_features: list[str] = CAT_CL_FEAT) -> ColumnTransformer:
    """
    Combines the numerical and categorical parallel processing branches.
    
    Args:
        num_features: List of numerical column names.
        cat_features: List of categorical column names.
    """
    return ColumnTransformer(
        transformers=[
            ('num', num_pipe(), NUM_CL_FEAT),
            ('cat', cat_pipe(), CAT_CL_FEAT)
        ]
    )


def projection_pipe(n_components: int = 2, 
                    n_neighbors: int = 20, 
                    random_state: int = 42) -> Pipeline:
    """
    Main pipeline that preprocesses data and runs UMAP dimensionality reduction.
    
    Args:
        n_components: The dimension of the space to embed into.
        n_neighbors: The size of local neighborhood used for manifold approximation.
        random_state: Seed for reproducibility.
    """
    return Pipeline(
        steps=[
            ('prep', prep_pipe()),
            ('umap', umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors, 
                min_dist=0.0,
                metric='cosine',
                random_state=random_state
            ))
        ]
    )


def linear_model_pipe() -> Pipeline:
    """
    Pipeline that preprocesses data and fits a Linear Regression model.
    """
    return Pipeline(
        steps=[
            ('prep', prep_pipe()),
            ('lin_reg', LinearRegression())
        ]
    )



