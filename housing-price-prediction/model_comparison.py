import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
# Local imports
from src.pipelines import linear_model_pipe
from config.general_settings import SEGMENTED_DATA_PATH



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
    test_rmse_per = ((mean_squared_error(y_test, y_pred)**0.5) / y_test.mean()) * 100 
    test_r2_per = r2_score(y_test, y_pred) * 100

    print(f"Test Error Rate:         {test_rmse_per:,.2f}%")
    print(f"Test Explained Variance: {test_r2_per:,.2f}%")

    # 4. Return predictions so we can plot them later if we want
    return y_pred


def main():
    print("Loading data...")

    segmented_data = pd.read_csv(SEGMENTED_DATA_PATH)
    
    X = segmented_data.drop(columns='price')
    y = segmented_data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # ==========================================
    # DICTIONARY OF MODELS TO TEST
    # ==========================================
    models_to_test = {
        "Linear Regression": linear_model_pipe()
    }
    
    # Dictionary to store predictions for plotting
    predictions = {}

    # Loop through all models automatically
    for model_name, pipeline in models_to_test.items():
        preds = evaluate_model(model_name, pipeline, X_train, y_train, X_test, y_test)
        predictions[model_name] = preds
    print("\nTraining and Evaluation Complete.")


if __name__ == "__main__":
    main()