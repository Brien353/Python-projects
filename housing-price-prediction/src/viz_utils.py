

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from scipy import stats
from src.data_utils import num_cat_cols_lst

# ==========================================
# Module Configuration
# ==========================================

sns.set_theme(context='notebook', style='whitegrid')


# ==========================================
# Plotting Functions
# ==========================================

def dist_plt(df: pd.DataFrame, var: str, show_kde: bool = False, 
             figsize: tuple = (10, 8), log_scale: bool = True) -> Axes:
    """
    Plots a histogram distribution of a specified DataFrame column.
    """
    mean_val = df[var].mean()
    median_val = df[var].median()

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(
        data=df,
        x=var,
        kde=show_kde,
        color='purple',
        log_scale=log_scale,
        ax=ax
    )
    
    ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(x=median_val, color='blue', linestyle='--', linewidth=2, label='Median')
    
    ax.set_xlabel(f'Variable: {var} - Log Scaled: {log_scale}')
    ax.set_ylabel('Frequency')
    
    
    if show_kde:
        ax.legend(['KDE Curve', 'Mean', 'Median'], loc='upper right')
    else:
        ax.legend(['Mean', 'Median'], loc='upper right')
        
    fig.tight_layout()
    return ax


def box_plt(df: pd.DataFrame, var: str, log_scale: bool = True, 
            figsize: tuple = (10, 8)) -> Axes:
    """
    Generates a box plot for a specified numerical variable to identify outliers.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x=var,
        log_scale=log_scale,
        color='purple',
        flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 8},
        ax=ax
    )
    
    ax.set_title(f'{var} - Box Plot (Log Scaled: {log_scale})')
    ax.set_xlabel(f'Variable: {var}')
    ax.set_ylabel('Frequency') 
    
    fig.tight_layout()
    return ax  


def corr_map_plt(corr_mat: pd.DataFrame, annotate: bool = True, 
                 figsize: tuple = (10, 8)) -> Axes:
    """
    Creates a heatmap representation of a correlation matrix.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data=corr_mat,
        cmap='icefire',
        annot=annotate,
        ax=ax
    )
    
    ax.set_title('Correlation Matrix Heat Map')
    fig.tight_layout()
    return ax


def clust_plot(X_proj: np.ndarray, labels: np.ndarray, 
               figsize: tuple = (10, 8)) -> Axes:
    """
    Plots a 2D UMAP projection colored by cluster labels (e.g., HDBSCAN).
    Assumes noise labels are represented by -1.
    """
    x_cord = X_proj[:, 0]
    y_cord = X_proj[:, 1]
    
    is_noise = (labels == -1)
    is_cluster = (labels != -1)
    cl_num = len(np.unique(labels[is_cluster]))

    fig, ax = plt.subplots(figsize=figsize)

    if np.any(is_noise):
        ax.scatter(
            x_cord[is_noise],
            y_cord[is_noise],
            c='red',
            label='Noise/Outliers',
            alpha=0.4,
            s=20,
            marker='x'
        )

    
    sns.scatterplot(
        x=x_cord[is_cluster],
        y=y_cord[is_cluster],
        hue=labels[is_cluster],
        palette='tab10',
        s=25,
        alpha=0.8,
        edgecolor='none',
        ax=ax
    )
    
    ax.set_title(f'UMAP-Projection ({cl_num} Clusters)', fontsize=14, pad=15)
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return ax


def residual_plt(pred: np.ndarray, residuals: np.ndarray, 
                 figsize: tuple = (10, 8)) -> Axes:
    """
    Plots model residuals against predicted values to check for heteroscedasticity.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(x=pred, y=residuals, alpha=0.5, ax=ax)
    
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Residuals vs. Predicted Prices')
    ax.set_xlabel('Predicted Price')
    ax.set_ylabel('Residual Error (Actual - Predicted)')
    
    fig.tight_layout()
    return ax



def sc_plot(data, x_lab, y_lab):
    ax = sns.scatterplot(
        data=data,
        x=x_lab,
        y=y_lab,
        color='red'
    )
    ax.set_title(f"{y_lab} vs {x_lab}")
    return ax



def pair_plot(df : pd.DataFrame) : 
    num_feat, _ =num_cat_cols_lst(df)
    df_num = df[num_feat]

    ax = sns.pairplot(df_num, corner= True, kind = 'reg')
    ax.fig.suptitle('Matriz de dispersión de variables numéricas')

    return ax


def multivaraite_distr_plot(df):
    num_cols , _ = num_cat_cols_lst(df)
    print(num_cols)
    N = len(df)
    p = len(num_cols)
    df_num = df[num_cols].to_numpy()
    x_bar = np.mean(df_num, axis = 0)
    S_inv = np.linalg.inv(np.cov(df_num, rowvar= False))
    diff = df_num - x_bar
    d2 = np.sum((diff @ S_inv) * diff, axis = 1)
    d2_sorted = np.sort(d2)
    probabilities = (np.arange(1, N + 1)-0.5) / N
    theoretical_quantiles = stats.chi2.ppf(probabilities, df = p)

    plt.figure(figsize=(7,6))
    plt.scatter(theoretical_quantiles, d2_sorted, alpha= 0.7, edgecolors= 'k', label = 'Observed Data')
    max_val = max(theoretical_quantiles.max(), d2_sorted.max())
    plt.plot([0,max_val],[0,max_val],'r--',lw = 2, label ='Multivariate Normal y= x')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Empirical squared Mahalanobis distance')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return d2

