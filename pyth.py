import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the LSTM model
loaded_model = load_model('lstm_autoencoder_model.h5')

# Load the CSV file
input_csv = './MIT-BIH Arrhythmia Database.csv'  # Replace with the actual path
data = pd.read_csv(input_csv)

# Inspect the data columns
print("Data columns: ", data.columns)

# Remove unnecessary columns like 'record' and 'type' for prediction
# (but we will keep them in the final output)
features = data.drop(['record', 'type'], axis=1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Reshape the data for the LSTM model (samples, time_steps, features)
features_reshaped = features_scaled.reshape(features_scaled.shape[0], 1, features_scaled.shape[1])

# Make predictions using the LSTM model
predictions = loaded_model.predict(features_reshaped)

# Assuming the target 'type' is categorical, encode it
label_encoder = LabelEncoder()
data['type_encoded'] = label_encoder.fit_transform(data['type'])

# Get the predicted labels based on model output
# Assuming your model outputs probabilities, we use np.argmax to get the predicted class
predicted_labels = label_encoder.inverse_transform([np.argmax(pred) for pred in predictions])

# Add the predicted labels to the original dataframe
data['predicted_type'] = predicted_labels

# Save the original data along with the predictions to a new CSV file
output_csv = './output_file2.csv'  # Replace with the desired output path
data.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")
