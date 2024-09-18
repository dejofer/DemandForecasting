import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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

# Target and features (predicting 'apple')
target = 'apple'
features = ['apple', 'banana', 'carrot', 'lettuce', 'tomato']

# Scaling data for better performance
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[features])

# Convert time series data into sequences for LSTM input
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(seq_length, len(data)):
        sequences.append(data[i-seq_length:i])
        labels.append(data[i, 0])  # We are predicting 'apple', which is the first column
    return np.array(sequences), np.array(labels)

# Set sequence length (number of previous weeks to consider)
seq_length = 10

# Prepare the data for LSTM
X, y = create_sequences(scaled_data, seq_length)

# Split the data into training and test sets
train_size = int(len(X) * 0.8)  # 80% for training
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Predicting a single value (demand for 'apple')
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Predict demand for this week
X_pred = np.array([scaled_data[-seq_length:]])  # The most recent sequence
y_pred_scaled = model.predict(X_pred)

# Inverse scaling to get the prediction back to original scale
y_pred = scaler.inverse_transform(np.concatenate((y_pred_scaled, np.zeros((1, len(features)-1))), axis=1))[:, 0]

# Print predicted demand
print(f"Predicted demand for {target} this week: {y_pred[0]}")

# Evaluation (optional)
# y_test_pred_scaled = model.predict(X_test)
# y_test_pred = scaler.inverse_transform(np.concatenate((y_test_pred_scaled, np.zeros((len(y_test_pred_scaled), len(features)-1))), axis=1))[:, 0]
# mae = np.mean(np.abs(y_test_pred - scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), len(features)-1))), axis=1))[:, 0]))
# print(f'Mean Absolute Error on test set: {mae}')
