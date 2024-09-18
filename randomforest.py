import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Define parameters
num_weeks = 1500
items = ['apple', 'banana', 'carrot', 'lettuce', 'tomato']
num_items = len(items)

# Initialize random seed for reproducibility
np.random.seed(42)

# Generate random sales data
data = np.random.randint(low=10, high=100, size=(num_weeks, num_items))

# Calculate the end date (most recent past Sunday)
end_date = datetime.now() - timedelta(days=datetime.now().weekday() + 1)

# Create a date range for the past 1500 weeks
date_rng = pd.date_range(end=end_date, periods=num_weeks, freq='W')

# Create a DataFrame
df = pd.DataFrame(data, columns=items, index=date_rng)
df.index.name = 'week'

print(df)

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
target = 'apple'  # WHAT YOU WANT TO FORECAST

X = current_data[features]
y = current_data[target]

# Training data: all weeks before the current week
train_data = df[df.index < current_data.index[0]]
X_train = train_data[features]
y_train = train_data[target]

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict demand for this week
X_pred = current_data[features]
y_pred = rf_model.predict(X_pred)

# Print predicted demand
print(f"Predicted demand for {target} this week: {y_pred[0]}")  # y_pred is a 1D array, get the first (and only) element

# Evaluation (optional)
# y_pred_train = rf_model.predict(X_train)
# mae_train = mean_absolute_error(y_train, y_pred_train)
# print(f'Mean Absolute Error on training set: {mae_train}')
