# improved_train.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def load_and_prepare_data(csv_path):
    """
    Load dataset and prepare features and target with cleaning.
    Adds NDVI trend feature and log-transforms the target.
    """
    df = pd.read_csv(csv_path)

    # Feature engineering: NDVI trend (placeholder calculation)
    df['early_season_ndvi'] = df['mean_ndvi'] * 0.9  # Replace with real data if possible
    df['late_season_ndvi'] = df['mean_ndvi'] * 1.1   # Replace with real data if possible
    df['ndvi_trend'] = df['late_season_ndvi'] - df['early_season_ndvi']

    # Select features and target
    features = df[['mean_ndvi', 'total_rainfall_mm', 'ndvi_trend']].replace([np.inf, -np.inf], np.nan)
    target = df['Yield_kg_per_hectare'].replace([np.inf, -np.inf], np.nan)

    # Filter out invalid rows
    valid_idx = features.notnull().all(axis=1) & target.notnull()

    features = features.loc[valid_idx]
    target = target.loc[valid_idx]

    # Optional: Keep only positive, realistic values
    features = features[(features['mean_ndvi'] > 0) & (features['total_rainfall_mm'] > 0)]
    target = target.loc[features.index]

    # Log-transform target to stabilize variance
    target_log = np.log1p(target)

    return features, target_log, target

def perform_hyperparameter_tuning(X_train, y_train):
    """
    Perform RandomizedSearchCV hyperparameter tuning for RandomForestRegressor.
    """
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'max_features': ['sqrt', 'log2', None],  # Corrected valid options
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, 
                                       cv=3, scoring='r2', random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test_log, y_test_raw):
    """
    Evaluate the model and print performance metrics.
    Also plots predicted vs true yields.
    """
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # inverse log-transform

    mae = mean_absolute_error(y_test_raw, y_pred)
    r2 = r2_score(y_test_raw, y_pred)
    rmse = mean_squared_error(y_test_raw, y_pred) ** 0.5

    print(f"Model Performance:")
    print(f"  MAE: {mae:.2f} kg/ha")
    print(f"  R²: {r2:.2f}")
    print(f"  RMSE: {rmse:.2f}")

    # Plot predicted vs true
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_raw, y=y_pred)
    plt.plot([y_test_raw.min(), y_test_raw.max()], [y_test_raw.min(), y_test_raw.max()], 'r--')
    plt.xlabel("True Yield (kg/ha)")
    plt.ylabel("Predicted Yield (kg/ha)")
    plt.title("Predicted vs True Crop Yields")
    plt.tight_layout()
    plt.show()

def main():
    # File path
    data_file = 'cleaned_with_gee_features.csv'

    # Load and prepare data
    features, target_log, target_raw = load_and_prepare_data(data_file)

    # Split dataset
    X_train, X_test, y_train, y_test_log = train_test_split(features, target_log, test_size=0.2, random_state=42)
    y_test_raw = target_raw.loc[y_test_log.index]

    # Hyperparameter tuning
    best_model = perform_hyperparameter_tuning(X_train, y_train)

    # Evaluate tuned model
    evaluate_model(best_model, X_test, y_test_log, y_test_raw)

    # Save the final model
    joblib.dump(best_model, 'yield_model_tuned.pkl')
    print("Saved tuned model as 'yield_model_tuned.pkl'")

if __name__ == '__main__':
    main()
