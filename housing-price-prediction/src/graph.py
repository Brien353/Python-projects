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
    


    
    

