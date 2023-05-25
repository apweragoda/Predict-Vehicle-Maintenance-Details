import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Load and preprocess the data
data = pd.read_csv('vehicle_maintenance_records.csv')

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

# Step 5: Get user input
user_input = []
for feature in selected_features:
    user_value = input(f"Enter the value for {feature}: ")
    user_input.append(user_value)

# Convert user input to numerical representation
user_input_encoded = []
for i, feature in enumerate(selected_features):
    if feature in label_encoders:
        user_input_encoded.append(
            label_encoders[feature].transform([user_input[i]])[0])
    else:
        user_input_encoded.append(user_input[i])

# Convert user input to numerical representation
user_input = np.array(user_input).reshape(1, -1)
for feature, encoder in label_encoders.items():
    if feature in selected_features:
        user_input[0, selected_features.index(feature)] = encoder.transform(
            [user_input[0, selected_features.index(feature)]])[0]

# Make prediction for user input
user_input_scaled = scaler.transform([user_input_encoded])
predicted_spare_parts = {}
spare_part_costs = {}
for feature in target_feature:
    prediction = model.predict(user_input_scaled)[0]
    if prediction > 0:
        predicted_spare_parts[feature] = 1
        spare_part_costs[feature] = prediction


user_input_scaled = scaler.transform(user_input)

# Step 6: Make predictions for user input data
user_prediction = model.predict(user_input_scaled)
predicted_total_cost = user_prediction[0]

# Provide the predicted total cost of spare parts to the user
print(f"Predicted Total Cost of Spare Parts: ${predicted_total_cost}")

# Step 7: Save the user input and predicted total cost to an output file
output_file = 'predicted_total_cost.txt'
with open(output_file, 'w') as f:
    f.write("User Input:\n")
    for feature, value in zip(selected_features, user_input[0]):
        f.write(f"{feature}: {value}\n")
    f.write(f"\nPredicted Total Cost of Spare Parts: {predicted_total_cost}\n")

print(f"User input and predicted total cost saved to {output_file}")


# Step 8: Save the model for future use
joblib.dump(model, 'spare_parts_model.joblib')
