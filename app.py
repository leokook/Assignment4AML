import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Load  data and model
best_model = joblib.load('best_model.joblib')



st.title('Breast Cancer Prediction App')

# Load  data that is saved as X_test.csv and y_test.csv
def load_data():
    X = pd.read_csv('X_test.csv')
    y = pd.read_csv('y_test.csv')
    return X, y

X, y = load_data()  # Implement this function to load our data


# Feature selection (use the same method as in our  training script)
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()

# Load the scaler
scaler = joblib.load('scaler.joblib')  # Load the same scaler as in training

# Create input fields for features
feature_inputs = {}
for feature in selected_features:
    feature_inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button('Predict'):
    input_data = np.array(list(feature_inputs.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_data)  # Make sure to use the same scaler as in training
    prediction = best_model.predict(input_scaled)
    st.write(f"Prediction: {'Malignant' if prediction[0] == 0 else 'Benign'}")