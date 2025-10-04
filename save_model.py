# save_model.py
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

print("Starting model training and saving process...")

# --- Load and prepare the data ---
df = pd.read_csv("synthetic_household_power.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

features = [
    'global_active_power', 'global_reactive_power', 'voltage',
    'global_intensity', 'sub_metering_1', 'sub_metering_2',
    'sub_metering_3', 'total_sub_metering', 'hour', 'weekday'
]
X = df[features]
y = df['target_next_hour_power']

# --- 1. Train and Save the Decision Tree Regressor ---
regressor_model = DecisionTreeRegressor(random_state=42)
regressor_model.fit(X, y) # Train on unscaled data
joblib.dump(regressor_model, 'power_forecasting_model.joblib')
print("✅ Regressor model saved.")

# --- 2. Create and Save the Scaler ---
# The scaler is only for the clustering model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.joblib')
print("✅ Scaler saved.")

# --- 3. Train and Save the K-Means Clustering Model ---
# K-Means works best on scaled data
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_model.fit(X_scaled)
joblib.dump(kmeans_model, 'kmeans_model.joblib')
print("✅ K-Means clustering model saved.")

print("\nAll models and the scaler have been saved successfully!")