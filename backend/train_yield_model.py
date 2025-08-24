import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import numpy as np

# Step 1: Load cleaned rice data
data_file = 'cleaned_production_data.csv'
df = pd.read_csv(data_file)

# Step 2: Generate synthetic NDVI and rainfall features (placeholder)
# Replace these with real GEE-based features later
np.random.seed(42)
df['mean_ndvi'] = np.clip(np.random.normal(loc=0.6, scale=0.1, size=len(df)), 0, 1)
df['total_rainfall_mm'] = np.clip(np.random.normal(loc=500, scale=100, size=len(df)), 0, None)

# Step 3: Prepare features and target
features = df[['mean_ndvi', 'total_rainfall_mm']]
target = df['Yield_kg_per_hectare']

# --- CLEAN DATA: Remove infinite and NaN values before training ---

# Replace infinite values with NaN
features = features.replace([np.inf, -np.inf], np.nan)
target = target.replace([np.inf, -np.inf], np.nan)

# Drop rows with any NaN in features or target
valid_idx = features.notnull().all(axis=1) & target.notnull()
features = features.loc[valid_idx]
target = target.loc[valid_idx]

# Convert to numpy arrays for sklearn
X = features.to_numpy()
y = target.to_numpy()

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:\n - MAE: {mae:.2f} kg/ha\n - R²: {r2:.2f}")

# Step 7: Save the trained model
model_filename = 'yield_model.pkl'
joblib.dump(model, model_filename)
print(f"Trained model saved to '{model_filename}'")
