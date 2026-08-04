import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes

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
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing the variable.
    var : str
        The column name to plot.
    show_kde : bool, default=False
        Whether to overlay a Kernel Density Estimate curve.
    figsize : tuple, default=(10, 8)
        Dimensions of the figure.
    log_scale : bool, default=True
        Whether to apply a logarithmic scale to the x-axis.
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object containing the plot.
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


X = hola