# train_and_compare_models.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

def load_data(filepath):
    """
    Load dataset, clean invalid values, feature engineering, and prepare target.
    Returns cleaned features DataFrame, log-transformed target Series, and original target Series.
    """
    df = pd.read_csv(filepath)
    
    # Feature engineering: NDVI trend (approximate - replace with real data if possible)
    df['early_season_ndvi'] = df['mean_ndvi'] * 0.9
    df['late_season_ndvi'] = df['mean_ndvi'] * 1.1
    df['ndvi_trend'] = df['late_season_ndvi'] - df['early_season_ndvi']

    features = df[['mean_ndvi', 'total_rainfall_mm', 'ndvi_trend']].replace([np.inf, -np.inf], np.nan)
    target = df['Yield_kg_per_hectare'].replace([np.inf, -np.inf], np.nan)

    # Remove rows with missing values in features or target
    valid_rows = features.notnull().all(axis=1) & target.notnull()
    features = features.loc[valid_rows]
    target = target.loc[valid_rows]

    # Optional filters for positive meaningful values
    features = features[(features['mean_ndvi'] > 0) & (features['total_rainfall_mm'] > 0)]
    target = target.loc[features.index]

    # Log-transform target to reduce skewness (add 1 to avoid log(0))
    target_log = np.log1p(target)

    return features, target_log, target

def tune_and_train_model(X_train, y_train):
    """
    Performs RandomizedSearchCV to find best hyperparameters for RandomForestRegressor.
    Returns the best trained RandomForest model.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'max_features': ['sqrt', 'log2', None],  # Corrected valid choices
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(rf, param_distributions=param_grid,
                                n_iter=10, cv=3, scoring='r2',
                                random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_

def evaluate_model(model, X_test, y_test_log, y_test_orig):
    """
    Evaluate model predictions with common regression metrics and plot predictions.
    """
    # Predict and inverse log-transform predictions
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # Calculate metrics on original scale
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    rmse = mean_squared_error(y_test_orig, y_pred) ** 0.5

    print(f"Model Performance:")
    print(f"  MAE : {mae:.2f} kg/ha")
    print(f"  R²  : {r2:.2f}")
    print(f"  RMSE: {rmse:.2f}")

    # Plot predicted vs true yields
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_orig, y=y_pred)
    plt.plot([y_test_orig.min(), y_test_orig.max()],
             [y_test_orig.min(), y_test_orig.max()], 'r--')
    plt.xlabel('True Yield (kg/ha)')
    plt.ylabel('Predicted Yield (kg/ha)')
    plt.title('Predicted vs True Crop Yields')
    plt.tight_layout()
    plt.show()

def main():
    DATA_FILE = 'cleaned_with_gee_features.csv'

    # Load and preprocess data
    features, target_log, target_orig = load_data(DATA_FILE)

    # Split into train and test sets
    X_train, X_test, y_train, y_test_log = train_test_split(features, target_log,
                                                            test_size=0.2, random_state=42)
    y_test_orig = target_orig.loc[y_test_log.index]

    # Train model with hyperparameter tuning
    best_model = tune_and_train_model(X_train, y_train)

    # Evaluate the trained model
    evaluate_model(best_model, X_test, y_test_log, y_test_orig)

    # Save the trained model
    joblib.dump(best_model, 'yield_model_tuned.pkl')
    print("Saved model as 'yield_model_tuned.pkl'")

if __name__ == '__main__':
    main()
