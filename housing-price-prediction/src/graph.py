import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


def dist_plt(df,var,kde_bol= False, figsize = (10,8),lg_scale_bol = True):
    sns.set_theme('notebook')
    mean_val = df[var].mean()
    median_val = df[var].median()

    plt.figure(figsize= figsize)
    sns.histplot(data= df,
                  x = f'{var}',
                  kde= kde_bol,
                  color = 'purple',
                  log_scale= lg_scale_bol)
    plt.axvline(x = mean_val, color = 'red',linestyle = '--', linewidth = 2)
    plt.axvline(x = median_val, color = 'blue', linestyle = '--', linewidth = 2)
    plt.xlabel(f'Variable : {var}- Log_scaled:{lg_scale_bol}')
    plt.legend(['KDE Curve Line', 'Mean', 'Median'], loc = 'upper right')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def box_plt(df,var, lg_scale_bol = True,figsize = (10,8)):
    sns.set_theme('notebook')
    plt.figure(figsize= figsize)
    sns.boxplot(
        data = df,
        x = f'{var}',
        log_scale= lg_scale_bol,
        color = 'purple',
        flierprops = {'marker':'o','markerfacecolor':'red','markersize':8}
    )
    plt.title(f'{var}-Box plot-Log_scaled:{lg_scale_bol}')
    plt.xlabel(f'Variable: {var}')
    plt.ylabel('Frequency') 
    plt.tight_layout()
    plt.show()  

def corr_map_plt(corr_mat,annot_bol = True,figsize = (10,8)):
    sns.set_theme('notebook')
    plt.figure(figsize=figsize)
    sns.heatmap(
        data= corr_mat,
        cbar= 'icefire',
        annot= annot_bol
    )
    plt.title('Correlation Matrix Heat Map')
    plt.tight_layout()
    plt.show()





def clust_plot(X_proj, labels):
    x_cord = X_proj[:, 0]
    y_cord = X_proj[:, 1]
    
    sns.set_theme(context='notebook', style='whitegrid')
    
    is_noise = (labels == -1)
    is_cluster = (labels != -1)
    
    cl_num = len(np.unique(labels[is_cluster]))

    if np.any(is_noise):
        plt.scatter(
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
        edgecolor='none'
    )
    # 6. Final Polish
    plt.title(f'UMAP-Projection ({cl_num} HDBSCAN Clusters)', fontsize=14, pad=15)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    # Moves the legend outside so it doesn't cover your data points
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()






