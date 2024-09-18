import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Generate example data (replace with your actual data loading)
num_weeks = 1500
items = ['apple', 'banana', 'carrot', 'lettuce', 'tomato']
num_items = len(items)

np.random.seed(42)
data = np.random.randint(low=10, high=100, size=(num_weeks, num_items))

end_date = pd.Timestamp.now().normalize() - pd.DateOffset(days=pd.Timestamp.now().normalize().dayofweek + 1)
date_rng = pd.date_range(end=end_date, periods=num_weeks, freq='W')

df = pd.DataFrame(data, columns=items, index=date_rng)
df.index.name = 'week'

# Prepare data for the current week
current_week = pd.Timestamp.now().week
current_year = pd.Timestamp.now().year

# Use isocalendar().week to get the week number
df['week'] = df.index.isocalendar().week
df['year'] = df.index.year

current_data = df[(df['year'] == current_year) & (df['week'] == current_week - 1)]  # Use previous week's data for prediction

# Features and target
features = ['apple', 'banana', 'carrot', 'lettuce', 'tomato']
target = 'apple'  # Predicting demand for apples as an example

X = current_data[features]
y = current_data[target]

# Training data: all weeks before the current week
train_data = df[df.index < current_data.index[0]]
X_train = train_data[features]
y_train = train_data[target]

# XGBoost model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict demand for this week
X_pred = current_data[features]
y_pred = xgb_model.predict(X_pred)

# Print predicted demand
print(f"Predicted demand for {target} this week: {y_pred[0]}")  # y_pred is a 1D array, get the first (and only) element

# Evaluation (optional)
# y_pred_train = xgb_model.predict(X_train)
# mae_train = mean_absolute_error(y_train, y_pred_train)
# print(f'Mean Absolute Error on training set: {mae_train}')
