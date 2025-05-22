import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import os
import sys
import textwrap

#Creating the path to the data set file
Data_path = r'C:\Users\DELL\Downloads\Economical Analysis Mexico\Info_empresarial.csv' 

#Creating the path  to the scian code file
scian_path = r'C:\Users\DELL\Downloads\Economical Analysis Mexico\Scian.xlsx' 
#Reading the data set of interest
dataset = pd.read_csv(Data_path, encoding= 'UTF-8',dtype={'giro': str, 'e_mail':str, 'telefono':str})
#Reading the scian file to translate scian into interpretable information
scian = pd.read_excel(scian_path, sheet_name="Claves SCIAN", header=2)

#Reading the columns of the data frame
columns_array = dataset.columns

#Reading the columns of the Scian file

columns_scian = scian.columns

#Counting the frequency of each Scian code  in order to generate a Scian distribution of the data set

scian_frequency_table = dataset.value_counts('scian').reset_index().set_axis(['scian', 'frequency'],axis=1)

#Joining both tables to traslate the scian into area

joined_tables = pd.merge(scian_frequency_table, scian,left_on='scian', right_on='CLAVE SCIAN',how='inner')

#Conserve the proper name

result = joined_tables[['DESCRIPCIÓN DE LA ACTIVIDAD','frequency']]

#Order the result table according frequency

result = result.sort_values('frequency', ascending= False)
result = result.drop_duplicates(subset=['DESCRIPCIÓN DE LA ACTIVIDAD', 'frequency'])



def create_bar_plot(data, title, x='frequency', y='DESCRIPCIÓN DE LA ACTIVIDAD', palette='coolwarm', max_label_length=40):
    # Create the figure with a specific size for better spacing
    plt.figure(figsize=(12, 8))

    # Create horizontal bar plot (swap x and y)
    ax = sns.barplot(data=data, y=y, x=x, palette=palette)

    # Title and labels with adjusted font sizes for readability
    plt.title(title, pad=20, fontsize=16)
    plt.xlabel('Prevalencia de la actividad económica', labelpad=15, fontsize=12)
    plt.ylabel('Descripción de la actividad económica', fontsize=12)

    # Wrap the y-axis labels with a maximum length of characters
    labels = ['\n'.join(textwrap.wrap(label.get_text(), max_label_length)) 
              for label in ax.get_yticklabels()]
    ax.set_yticklabels(labels, fontsize=8)

    # Add grid lines for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Remove spines for a cleaner look
    sns.despine(left=True, bottom=True)

    # Add data labels on each bar for clarity
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 50,  # Adjusted padding
                p.get_y() + p.get_height()/2., 
                f'{width:.2f}', 
                ha='left', 
                va='center', 
                fontsize=10,  # Adjusted font size for better visibility
                color='black')

    # Adjust layout to ensure labels and elements fit
    plt.tight_layout()

    # Display the plot
    plt.show()

# Segments: Top 5, Next 5, and so on until we reach the 30th row (in chunks of 5)
for i in range(0, 30, 5):
    create_bar_plot(result[i:i+5], 
                    title=f'Principales actividades económicas en México ({i+1} - {i+5})')

# Create plot for the last 5 elements (31-35)
create_bar_plot(result[30:35], 
                title='Últimas 5 actividades económicas en México')