import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# Mock function: Replace this with your actual data loading and preparation
def load_sequential_data():
    """
    Returns:
    X: numpy array, shape (num_samples, timesteps, num_features)
    y: numpy array, shape (num_samples,)
    
    Here, we generate dummy data for demonstration.
    """
    num_samples = 500
    timesteps = 10  # e.g., 10 weeks in growing season
    num_features = 3  # e.g., NDVI, rainfall, temperature
    
    # Random synthetic features
    X = np.random.rand(num_samples, timesteps, num_features)
    
    # Synthetic yields correlated somewhat with mean NDVI feature
    y = X[:, :, 0].mean(axis=1) * 1000 + np.random.normal(0, 50, num_samples)
    y = np.clip(y, 200, 5000)  # realistic yield limits
    
    return X, y

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # Load data
    X, y = load_sequential_data()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features per timestep and feature (reshape to 2D for scaler, then back)
    num_train_samples, timesteps, num_features = X_train.shape
    
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, num_features)
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_train = X_train_2d.reshape(num_train_samples, timesteps, num_features)
    
    num_test_samples = X_test.shape[0]
    X_test_2d = X_test.reshape(-1, num_features)
    X_test_2d = scaler.transform(X_test_2d)
    X_test = X_test_2d.reshape(num_test_samples, timesteps, num_features)
    
    # Build and train model
    model = build_lstm_model(input_shape=(timesteps, num_features))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                        validation_split=0.2, callbacks=[early_stop], verbose=1)
    
    # Predict and evaluate
    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MAE: {mae:.2f} kg/ha")
    print(f"Test R²: {r2:.2f}")
    
    # Save model and scaler
    model.save('lstm_yield_model.keras')
    joblib.dump(scaler, 'feature_scaler.pkl')
    print("Saved LSTM model and scaler")

if __name__ == '__main__':
    main()
