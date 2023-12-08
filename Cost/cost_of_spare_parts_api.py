import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Step 1: Load and preprocess the data
data = pd.read_csv('vehicle_maintenance_records_latest.csv')

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


# Endpoint for predicting the cost of spare parts


@app.route('/predict', methods=['POST'])
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


if __name__ == '__main__':
    app.run(debug=True)
