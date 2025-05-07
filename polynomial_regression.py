import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('MERGED_FINAL_DATASET.csv')
df = df.dropna().copy()

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Convert 'date' column to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Apply log transform to all features except interest rates and trading volumes
log_transform_cols = [col for col in df.columns if 'rate' not in col.lower() and 'volume' not in col.lower()]
df[log_transform_cols] = np.log(df[log_transform_cols])

# Replace infinite values and fill missing data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)

# Define target variable
target_col = 'cushing_crude_oil_price'

# Columns to exclude from features

exclude_cols = [
    'natural_gas_futures_contract_1_price',
    'natural_gas_futures_contract_2_price',
    'natural_gas_futures_contract_3_price',
    'natural_gas_futures_contract_4_price',
    'cushing_crude_futures_contract_1_price',
    'cushing_crude_futures_contract_2_price',
    'cushing_crude_futures_contract_3_price',
    'cushing_crude_futures_contract_4_price'
]

# Define features (drop target and excluded columns)
X = df.drop(columns=[target_col] + exclude_cols)
y = df[target_col]

# Split dataset (90% train, 10% test)
split_idx = int(0.9 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial feature transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train polynomial regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
intercept = model.intercept_
weights = model.coef_

# Output
print("Mean Squared Error (MSE):", mse)
print("Intercept:", intercept)
#print("First 10 Weights:", weights[:-1])
