import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import sys
import os

#Definir la ruta de las bases de datos necesarias para el análisis
def contar_nacimientos_zac_1(path_ano):
    # Cargando la base de datos de natalidad
    Nat_ano = pd.read_csv(path_ano,low_memory=False)
    
    # Lista de columnas de los registros
    columnas = Nat_ano.columns
    
    # Filtrando por entidad de nacimiento en Zacatecas
    Nat_ano_Zac = Nat_ano[Nat_ano['entidad_nacimiento'] == 'ZACATECAS']
    Nat_ano_Zac = Nat_ano_Zac[Nat_ano_Zac['entidad_residencia_madre'] == 'ZACATECAS']
    
    # Guardar el número de nacimientos en el año específico
    nat_año_zac = len(Nat_ano_Zac)
    
    return nat_año_zac


def contar_nacimientos_zac_2(path_ano):
    # Cargando la base de datos de natalidad
    Nat_ano = pd.read_csv(path_ano,low_memory=False)
    
    # Lista de columnas de los registros
    columnas = Nat_ano.columns
    
    # Filtrando por entidad de nacimiento en Zacatecas
    Nat_ano_Zac = Nat_ano[Nat_ano['ENT_NAC'] == 32]
    Nat_ano_Zac = Nat_ano_Zac[Nat_ano_Zac['ENT_RES'] ==32]
    
    # Guardar el número de nacimientos en el año específico
    nat_año_zac = len(Nat_ano_Zac)
    
    return nat_año_zac



def contar_nacimientos_zac_3(path_ano):
    # Cargando la base de datos de natalidad
    Nat_ano = pd.read_csv(path_ano,low_memory=False)
    
    # Lista de columnas de los registros
    columnas = Nat_ano.columns
    
    # Filtrando por entidad de nacimiento en Zacatecas
    Nat_ano_Zac = Nat_ano[Nat_ano['ENTIDADFEDERATIVAPARTO'] == 32]
    Nat_ano_Zac = Nat_ano_Zac[Nat_ano_Zac['ENTIDADRESIDENCIA'] ==32]
    
    # Guardar el número de nacimientos en el año específico
    nat_año_zac = len(Nat_ano_Zac)
    
    return nat_año_zac


def generar_nacimientos_zacatecas(paths_1, paths_2, paths_3, ruta_salida_csv):
    # Procesar datos con las funciones respectivas
    nacimientos_1 = [contar_nacimientos_zac_1(path) for path in paths_1]
    nacimientos_2 = [contar_nacimientos_zac_2(path) for path in paths_2]
    nacimientos_3 = [contar_nacimientos_zac_3(path) for path in paths_3]

    # Unir todos los resultados
    nacimientos = nacimientos_1 + nacimientos_2 + nacimientos_3
    años = list(range(2008, 2024))

    # Crear y guardar el DataFrame
    df = pd.DataFrame({
        "Año": años,
        "Nacimientos": nacimientos
    })
    df.to_csv(ruta_salida_csv, index=False)


# Definiendo la ruta del archivo para los años 2008-2016 los cuales trabajan correctamente con la primera función de filtrado de datos
path_2008 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2008.csv'
path_2009 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2009.csv'
path_2010 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2010.csv'
path_2011 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2011.csv'
path_2012 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2012.csv'
path_2013 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2013.csv'
path_2014 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2014.csv'
path_2015 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2015.csv'
path_2016 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2016.csv'

#Definiendo la ruta del archivo para los años 2017-2019 los cuales trabajan correctamente con la segunda función de filtrado de datos, esto debido a que los nombres de las columnas de interés han cambiado a lo largo del tiempo
path_2017 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2017.csv'
path_2018 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2018.csv'
path_2019 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2019.csv'


# Definiendo la ruta del archivo para los años 2020-2023 los cuales trabajan correctamente con la tercera función de filtrado de datos, esto debido a que los nombres de las columnas de interés han cambiado a lo largo del tiempo
path_2020 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2020.csv'
path_2021 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2021.csv'
path_2022 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2022.csv'
path_2023 = r'C:\Users\DELL\Downloads\Natalidad\Nat_2023.csv'


# Llamando a la función para contar los nacimientos en Zacatecas en 2008-2016

# Rutas a los archivos
paths_1 = [path_2008, path_2009, path_2010, path_2011, path_2012, path_2013, path_2014, path_2015, path_2016]
paths_2 = [path_2017, path_2018, path_2019]
paths_3 = [path_2020, path_2021, path_2022, path_2023]

# Ruta de salida del CSV
ruta_csv = r"C:\Users\DELL\Downloads\Análisis de la Natalidad en Zacatecas (2008–2023)\Resumen\nacimientos_zacatecas_acumulado.csv"

#llamando a la función que procesa los datos

#generar_nacimientos_zacatecas(paths_1, paths_2, paths_3, ruta_csv)


#Extrayendo información util de la información procesada
Nacimientos_zac = pd.read_csv(ruta_csv)


#Año con más nacimientos
Nacimientos_zac_desc= Nacimientos_zac.sort_values(by='Nacimientos',ascending=False)
Año_con_mas_nacimientos = Nacimientos_zac_desc.iloc[0]["Año"]

#Año con menos nacimientos
Nacimientos_zac_asc = Nacimientos_zac.sort_values(by='Nacimientos',ascending=True)
Año_con_mas_nacimientos = Nacimientos_zac_asc.iloc[0]["Año"]


#Calcular la media de nacimientos en el periodo 2008-20023
media_nacimientos = Nacimientos_zac['Nacimientos'].mean()


#Tendencia de los nacimientos

#Comportamiento general

#Graficar el comportamiento del número de nacimientos a través de los años 2008-2023

# Estilo profesional
sns.set_theme(style="whitegrid")

# Crear figura
plt.figure(figsize=(12, 6))
sns.lineplot(data=Nacimientos_zac, x="Año", y="Nacimientos", marker="o", linewidth=2, label="Nacimientos por año")

# Línea roja del promedio
plt.axhline(media_nacimientos, color='red', linestyle='--', linewidth=2, label=f"Promedio (2008–2023): {int(media_nacimientos)} nacimientos")

# Títulos y etiquetas con mejor presentación
plt.title("Evolución Anual de Nacimientos en Zacatecas (2008–2023)", fontsize=16)
plt.xlabel("Año", fontsize=12)
plt.ylabel("Cantidad de Nacimientos", fontsize=12)

# Personalizar ejes y leyenda
plt.xticks(rotation=45)
plt.legend(loc="upper right", fontsize=10)
plt.grid(True)
plt.tight_layout()

# Mostrar gráfica
plt.show()

#Identificar la tendencia de los nacimientos en Zacatecas a través de los años

#La pendiente en una regresión lineal nos dice cómo cambia una variable (por ejemplo, nacimientos) en relación con otra (por ejemplo, el año).

#Si la pendiente es positiva ➜ los nacimientos aumentan con el tiempo.

#Si es negativa ➜ los nacimientos disminuyen.

#Si es cercana a 0 ➜ hay poca o ninguna tendencia
# Calcular la pendiente

resultado = linregress(Nacimientos_zac["Año"], Nacimientos_zac["Nacimientos"])
pendiente = resultado.slope
pendiente_redondeada = round(pendiente, 2)

# Asignar color y descripción según la pendiente
if pendiente > 0:
    color_linea = "green"
    tendencia_texto = f"Tendencia positiva (+{pendiente_redondeada} nacimientos/año)"
elif pendiente < 0:
    color_linea = "red"
    tendencia_texto = f"Tendencia negativa ({pendiente_redondeada} nacimientos/año)"
else:
    color_linea = "purple"
    tendencia_texto = "Tendencia estable (sin cambio)"

# Estilo profesional
sns.set_theme(style="whitegrid")

# Crear la gráfica
plt.figure(figsize=(12, 6))
sns.regplot(
    data=Nacimientos_zac,
    x="Año",
    y="Nacimientos",
    marker="o",
    scatter_kws={"s": 60, "color": "black"},
    line_kws={"color": color_linea, "label": tendencia_texto},
)

# Títulos y etiquetas
plt.title("Tendencia de Nacimientos en Zacatecas (2008–2023)", fontsize=16)
plt.xlabel("Año", fontsize=12)
plt.ylabel("Cantidad de Nacimientos", fontsize=12)
plt.xticks(rotation=45)

# Leyenda clara
plt.legend(loc="upper right", fontsize=10)

# Estética final
plt.grid(True)
plt.tight_layout()
plt.show()


#¿En cuántos años los nacimientos estuvieron por debajo de la media?
Años_por_debajo_media= Nacimientos_zac[Nacimientos_zac['Nacimientos']< media_nacimientos]


#¿En cuántos años los nacimientos estuvieron por encima de la media?
Años_por_debajo_media= Nacimientos_zac[Nacimientos_zac['Nacimientos']> media_nacimientos]


#Se refiere a calcular cuánto cambiaron los nacimientos cada año respecto al anterior, en forma de porcentaje.

Nacimientos_zac["Cambio %"] = Nacimientos_zac["Nacimientos"].pct_change() * 100

# Estilo profesional
sns.set_theme(style="whitegrid")

# Crear la gráfica
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=Nacimientos_zac,
    x="Año",
    y="Cambio %",
    marker="o",
    color="orange",
    linewidth=2,
    label="Cambio porcentual anual"
)

# Línea de referencia en 0%
plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, label="Sin cambio (0%)")

# Títulos y etiquetas
plt.title("Variación Porcentual Anual de Nacimientos en Zacatecas (2008–2023)", fontsize=16)
plt.xlabel("Año", fontsize=12)
plt.ylabel("Cambio porcentual respecto al año anterior (%)", fontsize=12)

# Ejes y leyenda
plt.xticks(rotation=45)
plt.legend(loc="upper right", fontsize=10)
plt.grid(True)
plt.tight_layout()

# Mostrar la gráfica
plt.show()

#Hubo caídas o aumentos abruptos en ciertos años
q=Nacimientos_zac[Nacimientos_zac["Cambio %"].abs() > 5]  # Cambios mayores al 5%
print(q)

#¿En qué años hubo una disminución en comparación al año anterior?

w= Nacimientos_zac[Nacimientos_zac["Nacimientos"].diff() < 0]
print(w)





