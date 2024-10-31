import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
import pandas as pd
import matplotlib.pyplot as plt

# Data Import
dataset = pd.read_csv('final_dataset.csv')
print(f"Example observations:\n{dataset.head()}")

# Data Preprocessing
features = ['warning_proportion', 'error_proportion', 'avg_cpu_usage_percent', 'avg_memory_usage_percent', 'avg_latency_milliseconds', 'avg_disk_usage_percent', 'hour']
X = dataset[features]

# Split data: 70% train, 10% validation, 20% test
X_train_val, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_train_val, test_size=0.125, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape data to 3D for LSTM [samples, timesteps, features]
timesteps = 10
def create_sequences(data, timesteps):
    return np.array([data[i:i+timesteps] for i in range(len(data)-timesteps+1)])

X_train_scaled = create_sequences(X_train_scaled, timesteps)
X_val_scaled = create_sequences(X_val_scaled, timesteps)
X_test_scaled = create_sequences(X_test_scaled, timesteps)

# Define the LSTM autoencoder architecture
input_dim = X_train_scaled.shape[2]
hidden_state_size = 6

input_layer = Input(shape=(timesteps, input_dim))
encoder = LSTM(hidden_state_size, activation='relu', return_sequences=False)(input_layer)
repeat_vector = RepeatVector(timesteps)(encoder)
decoder = LSTM(input_dim, activation='linear', return_sequences=True)(repeat_vector)

autoencoder = Model(inputs=input_layer, outputs=decoder)

# Callbacks for early stopping and choosing best model from training history
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Compile and train the model
autoencoder.compile(optimizer='adam', loss=Huber())
history = autoencoder.fit(X_train_scaled, X_train_scaled, 
                          epochs=200, 
                          batch_size=32, 
                          shuffle=True, 
                          validation_data=(X_val_scaled, X_val_scaled),
                          callbacks=callbacks,
                          verbose=1)

# Function to detect anomalies
def detect_anomalies(model, data, threshold):
    reconstructions = model.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=(1, 2))
    return mse, mse > threshold

# Calculate the threshold using the validation set
reconstructions_val = autoencoder.predict(X_val_scaled)
mse_val = np.mean(np.power(X_val_scaled - reconstructions_val, 2), axis=(1, 2))

test_loss = autoencoder.evaluate(X_test_scaled, X_test_scaled, verbose=0)
print(f"Test loss: {test_loss}")

# Visualize the reconstruction error
plt.figure(figsize=(10, 6))
plt.hist(mse_val, bins=50)
plt.xlabel("Reconstruction error")
plt.ylabel("Number of samples")
plt.title("Reconstruction error on validation set")
plt.show()

# Set the threshold to the 97.5th percentile of the validation set reconstruction error
threshold = np.percentile(mse_val, 97.5)

# Detect anomalies in the test set
mse_test, anomalies = detect_anomalies(autoencoder, X_test_scaled, threshold)

print(f"Number of observations in the test set: {len(X_test_scaled)}")
print(f"Number of anomalies detected: {np.sum(anomalies)}")
print(f"Percentage of anomalies: {np.mean(anomalies)*100:.2f}%")

# Print 5 of the anomalies with column names
anomalous_indices = np.where(anomalies)[0]
print("Indices of anomalies:", anomalous_indices[:5])
print("Anomalous data points:")

# Extract the original data for the anomalies
original_anomalous_data = X_test.iloc[anomalous_indices[:np.sum(anomalies)]]
print(original_anomalous_data)
original_anomalous_data.to_csv('original_anomalous_data.csv', index=False)

# Visualize the reconstruction error on the test set
plt.figure(figsize=(10, 6))
plt.hist(mse_test, bins=50)
plt.xlabel("Reconstruction error")
plt.ylabel("Number of samples")
plt.title("Reconstruction error on test set")
plt.show()

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
