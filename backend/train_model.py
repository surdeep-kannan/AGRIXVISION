# backend/train_model.py
# This script reads the final training dataset and creates the prediction model.

import pandas as pd
import xgboost as xgb
import joblib
import os
import numpy as np # Import numpy to handle infinite values
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_yield_model():
    """
    Loads the enriched training dataset, trains an XGBoost regression model,
    evaluates its performance, and saves it to a file.
    """
    print("--- Starting Model Training ---")

    # --- 1. Load the Final Training Dataset ---
    input_filename = 'training_dataset.csv'
    if not os.path.exists(input_filename):
        print(f"--- FATAL ERROR: Training data file not found! ---")
        print(f"Please run 'enrich_data.py' first to create '{input_filename}'.")
        return

    try:
        df = pd.read_csv(input_filename)
        print(f"Successfully loaded '{input_filename}' with {len(df)} records.")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # --- 2. Prepare and Clean the Data for Training ---
    features = ['mean_ndvi', 'total_rainfall_mm']
    target = 'Yield_kg_per_hectare'

    if not all(col in df.columns for col in features + [target]):
        print(f"--- ERROR: The training CSV is missing required columns. ---")
        return

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    original_rows = len(df)
    df.dropna(subset=features + [target], inplace=True)
    cleaned_rows = len(df)
    if original_rows > cleaned_rows:
        print(f"Cleaned data: Removed {original_rows - cleaned_rows} rows with invalid data.")

    X = df[features]
    y = df[target]
    print("Data prepared for training.")

    # --- 3. Split Data into Training and Testing Sets ---
    # We'll use 80% of the data for training and 20% for testing.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training records and {len(X_test)} testing records.")

    # --- 4. Train the XGBoost Model ---
    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8
    )

    # The model is trained ONLY on the training data
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 5. Evaluate the Model's Performance ---
    print("\n--- Evaluating Model Performance ---")
    # Make predictions on the unseen test data
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f} kg/hectare")
    print(f"R-squared (R²): {r2:.2f}")
    print("(R² closer to 1.0 is better, MAE closer to 0 is better)")

    # --- 6. Visualize the Results ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Yield (kg/hectare)")
    plt.ylabel("Predicted Yield (kg/hectare)")
    plt.title("Model Performance: Actual vs. Predicted Yield")
    plt.savefig('model_performance.png')
    print("\nPerformance plot saved as 'model_performance.png'")


    # --- 7. Save the Trained Model ---
    output_filename = 'yield_model.pkl'
    joblib.dump(model, output_filename)

    print(f"\n--- SUCCESS ---")
    print(f"Model was successfully trained and saved as '{output_filename}'")
    print("You are now ready to start the main FastAPI server!")
    
# --- Run the Script ---
if __name__ == '__main__':
    train_yield_model()
