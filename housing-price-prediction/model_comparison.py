"""
src/models/train.py
Trains and evaluates regression models using cross-validation and test set evaluation.
Saves winning model artifacts for deployment pipelines.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from config.general_settings import SEGMENTED_DATA_PATH
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from src.pipelines import linear_model_pipe

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Runs CV, fits the model, and calculates test metrics for a single model."""
    print(f"\n{'='*40}")
    print(f"Evaluating: {name}")
    print(f"{'='*40}")
    
    # 1. Cross-validation
    print("Running cross-validation...")
    model_scores = cross_val_score(
        model, X_train, y_train, 
        cv=10, n_jobs=-1, scoring='neg_root_mean_squared_error'
    )
    model_r2 = cross_val_score(
        model, X_train, y_train, 
        cv=10, n_jobs=-1, scoring='r2'
    )

    cv_rmse_per = (-model_scores.mean() / y_train.mean()) * 100
    print(f"CV Expected Error Rate:  {cv_rmse_per:,.2f}%")
    print(f"CV Explained Variance:   {model_r2.mean()*100:,.2f}%")

    # 2. Final Model Fit
    print("Fitting model on full training data...")
    model.fit(X_train, y_train)
    
    # 3. Test Set Evaluation
    y_pred = model.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred)**0.5
    test_rmse_per = (test_rmse / y_test.mean()) * 100 
    test_r2_per = r2_score(y_test, y_pred) * 100

    print(f"Test Error Rate:         {test_rmse_per:,.2f}%")
    print(f"Test Explained Variance: {test_r2_per:,.2f}%")

    metrics = {
        "model_name": name,
        "cv_rmse_percentage": float(cv_rmse_per),
        "cv_r2_percentage": float(model_r2.mean() * 100),
        "test_rmse": float(test_rmse),
        "test_rmse_percentage": float(test_rmse_per),
        "test_r2_percentage": float(test_r2_per)
    }

    return model, y_pred, metrics


def main():
    print("Loading data...")
    segmented_data = pd.read_csv(SEGMENTED_DATA_PATH)
    
    X = segmented_data.drop(columns='price')
    y = segmented_data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # DICTIONARY OF MODELS TO TEST
    models_to_test = {
        "Linear Regression": linear_model_pipe()
    }
    
    best_model = None
    best_r2 = -float("inf")
    all_metrics = {}

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, pipeline in models_to_test.items():
        fitted_model, _ , metrics = evaluate_model(
            model_name, pipeline, X_train, y_train, X_test, y_test
        )
        all_metrics[model_name] = metrics
        
        # Track Champion Model based on R2 Score
        if metrics["test_r2_percentage"] > best_r2:
            best_r2 = metrics["test_r2_percentage"]
            best_model = fitted_model

    # Save model artifact and metrics JSON for CI/CD checks
    joblib.dump(best_model, ARTIFACTS_DIR / "champion_model.joblib")
    with open(ARTIFACTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved Champion model and metrics to {ARTIFACTS_DIR}")
    print("Training and Evaluation Complete.")


if __name__ == "__main__":
    main()