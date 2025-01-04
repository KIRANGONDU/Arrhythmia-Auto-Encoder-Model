import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

# Load your dataset
data = pd.read_csv('MIT-BIH Arrhythmia Database.csv')
# Data Preparation
# Extract features (excluding 'record' and 'type' columns) and labels (type column)
features = data.drop(['record', 'type'], axis=1).values
labels = data['type'].values

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Encode the labels (convert 'N', 'VEB', etc., to numerical values)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Reshape the features for LSTM input (samples, time steps, features)
# Here, we treat each row as a time step for the sake of the model structure
features_scaled_reshaped = np.expand_dims(features_scaled, axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_scaled_reshaped, labels_encoded, test_size=0.2, random_state=42)

# LSTM Autoencoder Model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    RepeatVector(1),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X_train.shape[2])),
    # Add a classification head to predict the label (type)
    Dense(64, activation='relu'),
    Dense(len(np.unique(labels_encoded)), activation='softmax')  # Softmax for classification
])

# Compile the model with loss for both reconstruction and classification
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Plot training & validation accuracy values
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

print(X_test[0])
# Now let's predict one sample from the test data
sample = X_test[0].reshape(1, 1, X_train.shape[2])
prediction = model.predict(sample)

# Print the original and predicted type for the sample
predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
original_label = label_encoder.inverse_transform([y_test[0]])

(original_label[0], predicted_label[0])  # Format (Original, Predicted)

print(original_label[0])
print(predicted_label[0])

from tensorflow.keras.models import load_model

# Save the model
model.save('lstm_autoencoder_model.h5')
# Load the model
loaded_model = load_model('lstm_autoencoder_model.h5')
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Your custom input data as a list
custom_input = [267, 258, 0.053857278, 0.121722682, 0.494739949, -0.459246103, 0.050227634, 25, 5, 60, 30, 0.050227634, 0.084112393, 0.292198805, 0.493704883, -0.039848932, 267, 258, 0.001155876, 0.381689392, -0.027546461, -0.429080123, -0.027546461, 8, 2, 22, 12, -0.027546461, -0.063113783, -0.174157691, -0.241601522, -0.367118162]

# Normalize the input
scaler = MinMaxScaler()
custom_input_scaled = scaler.fit_transform(np.array(custom_input).reshape(1, -1))

# Reshape for LSTM input (1 sample, 1 time step, number of features)
custom_input_reshaped = custom_input_scaled.reshape(1, 1, -1)
# Make a prediction
prediction = loaded_model.predict(custom_input_reshaped)

# Get the predicted label
predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
print(f'Predicted label: {predicted_label[0]}')

predictions = []

for sample in X_test:
    # Make sure to reshape sample correctly
    sample_reshaped = sample.reshape(1, 1, X_train.shape[2])  # (1 sample, 1 time step, features)
    
    # Predict
    prediction = loaded_model.predict(sample_reshaped)
    
    # Get the predicted label
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    predictions.append(predicted_label[0])

# Now predictions contain the predicted labels for all samples in X_test
print(predictions)