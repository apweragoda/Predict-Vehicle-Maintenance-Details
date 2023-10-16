import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from flask import Flask, jsonify, request
import json

app = Flask(__name__)

# Step 1: Load and preprocess the data
data = pd.read_csv('vehicle_maintenance_records_updated.csv')

# Step 2: Feature engineering and data preparation
selected_features = ['vehicle_type', 'brand',
                     'model', 'engine_type', 'make_year', 'mileage']
target_feature = 'cost'

# Select relevant features and target variable
df = data[selected_features + [target_feature]].copy()

# Handle missing values if any
df.dropna(inplace=True)

# Convert categorical variables to numerical representation
label_encoders = {}
for feature in selected_features:
    if df[feature].dtype == 'object':
        label_encoders[feature] = LabelEncoder()
        df[feature] = label_encoders[feature].fit_transform(df[feature])

# Split the data into training and testing sets
X = df[selected_features]
y = df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 3: Train a machine learning model
# Apply feature scaling to normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Endpoint for predicting the spare parts and remaining mileage
@app.route('/predictCost', methods=['POST'])
def predict_spare_parts_cost():
    # Get the user input from the request body as JSON
    user_input = request.json

    # Check if user input is empty
    if not user_input:
        return jsonify({'error': 'No data provided'})

    try:
        # Convert user input to a dataframe
        user_input_df = pd.DataFrame([user_input], columns=selected_features)

        # Convert categorical variables to numerical representation
        for feature, encoder in label_encoders.items():
            if feature in selected_features:
                user_input_df[feature] = encoder.transform(
                    user_input_df[feature])

        # Apply feature scaling to normalize the data
        user_input_scaled = scaler.transform(user_input_df)

        # Make prediction for user input
        predicted_total_cost = model.predict(user_input_scaled)[0]

        # Prepare the response
        response = {
            'predicted_total_cost': predicted_total_cost
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/predict', methods=['POST'])
def predict_spare_parts():
    # Get the user input from the request body as JSON
    user_input = request.get_json()
    # Step 1: Load and preprocess the data
    data = pd.read_csv('vehicle_maintenance_records_updated.csv')

    # Step 2: Feature engineering and data preparation
    selected_features = ['vehicle_type', 'brand',
                        'model', 'engine_type', 'make_year', 'mileage']
    target_feature = ['mileage_range', 'oil_filter', 'engine_oil', 'washer_plug_drain', 'dust_and_pollen_filter', 'air_clean_filter', 'fuel_filter', 'spark_plug',
                    'brake_fluid', 'brake_and_clutch_oil', 'transmission_fluid', 'brake_pads', 'clutch', 'coolant']

    # Select relevant features and target variable
    df = data[selected_features + target_feature].copy()

    # Handle missing values if any
    df.dropna(inplace=True)

    # Convert categorical variables to numerical representation
    label_encoders = {}
    for feature in selected_features:
        if df[feature].dtype == 'object':
            label_encoders[feature] = LabelEncoder()
            df[feature] = label_encoders[feature].fit_transform(df[feature])

    # Prepare the data for training
    X = df.drop(columns=target_feature)
    y = df[target_feature]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Step 3: Train a machine learning model
    # Apply feature scaling to normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Random Forest classifier for each spare part
    models = {}
    for feature in target_feature:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train[feature])
        models[feature] = model

    # Step 4: Evaluate the models
    y_pred = {}
    for feature, model in models.items():
        y_pred[feature] = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test[feature], y_pred[feature])
        print(f"Accuracy for {feature}: {accuracy}")
    # Check if user input is empty
    if not user_input:
        return jsonify({'error': 'No data provided'})

    try:
        # Convert user input to a dataframe
        user_input_df = pd.DataFrame([user_input], columns=selected_features)

        # Convert categorical variables to numerical representation
        for feature, encoder in label_encoders.items():
            if feature in selected_features:
                user_input_df[feature] = encoder.transform(
                    user_input_df[feature])

        # Apply feature scaling to normalize the data
        user_input_scaled = scaler.transform(user_input_df)

        # Make prediction for user input
        predicted_spare_parts = {}
        mileage_range = {}
        remaining_mileage = {}
        user_input_mileage = {}

        for feature, model in models.items():
            prediction = model.predict(user_input_scaled)[0]
            if prediction == 1:
                predicted_spare_parts[feature] = bool(prediction)
                user_input_mileage = int(user_input_df['mileage'])

                remaining_mileage[feature] = int(
                    df[df[feature] == 1]['mileage'].mean()) - int(user_input_mileage)
                if user_input_mileage < remaining_mileage[feature]:
                    mileage_range[feature] = int(
                        df[df[feature] == 1]['mileage_range'].mean()) - int(remaining_mileage[feature])
                if user_input_mileage >= remaining_mileage[feature]:
                    mileage_range[feature] = int(
                        df[df[feature] == 1]['mileage_range'].mean()) + int(remaining_mileage[feature])
        # Prepare the response
        response = {
            'predicted_spare_parts': predicted_spare_parts,
            'remaining_mileage': mileage_range,
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
