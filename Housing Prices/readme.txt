
# Advanced Real Estate Valuation Model
[![Regression](https://img.shields.io/badge/Task-Regression-green.svg)]()
[![Polynomial-Regression](https://img.shields.io/badge/Model-Polynomial-blue.svg)]()

## 🚀 Project Overview
Predicting property values using the Housing Dataset. This project demonstrates advanced data cleaning techniques, non-linear feature mapping, and rigorous residual diagnostics to ensure model reliability.

## 🧠 Key Data Science Strategies
* **Log-Target Transformation:** Applied `np.log1p` to the price variable to normalize the distribution and minimize relative error (MAPE) rather than absolute error.
* **Polynomial Mapping:** Utilized 2nd-degree polynomial features to capture non-linear interactions between area, stories, and amenities.
* **Pipeline Architecture:** Used `ColumnTransformer` to seamlessly handle one-hot encoding for categorical status and standard scaling for numerical features.

## 📉 Diagnostic Analysis (The "Residual" Check)
Unlike standard regression projects, this model includes a full diagnostic sweep:
* **Homoscedasticity:** Confirmed stable variance across all price points via Log-transformation.
* **$R^2$ Score:** 0.612 (A 61% improvement over the naive baseline).



## 💡 Results
The model reduced the baseline **RMSE from 0.346 to 0.215**, achieving a reliable error margin of ~32% on the test set—a significant achievement given the volatility of real estate markets.