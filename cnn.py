import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Load your dataset (adjust file path and format accordingly)
data = pd.read_csv('bill_authentication.csv')

# Separate features and labels
X = data[['variance', 'skewness', 'curtosis', 'entropy']].values
y = data['class'].values

# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CNN model
model = keras.Sequential([
    layers.Input(shape=(4,)),  # Assuming you have 4 features
    layers.Reshape(target_shape=(4, 1)),
    layers.Conv1D(32, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# Make predictions
predictions = model.predict(X_test)

# You can further analyze the predictions as needed for your specific task.
