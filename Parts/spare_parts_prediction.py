import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load and preprocess the data
data = pd.read_csv('vehicle_maintenance_records_updated.csv')

# Step 2: Feature engineering and data preparation
selected_features = ['vehicle_type', 'brand',
                     'model', 'engine_type', 'make_year', 'mileage']

target_feature = ['oil_filter', 'engine_oil', 'washer_plug_drain', 'dust_and_pollen_filter', 'air_clean_filter',
                  'fuel_filter', 'spark_plug',
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

# Step 5: Get user input for feature values
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

# Make prediction for user input
user_input_scaled = scaler.transform([user_input_encoded])
predicted_spare_parts = {}
remaining_mileage = {}
for feature, model in models.items():
    prediction = model.predict(user_input_scaled)[0]
    if prediction == 1:
        predicted_spare_parts[feature] = prediction
        remaining_mileage[feature] = df[df[feature] == 1]['mileage'].mean(
        ) - int(user_input[selected_features.index('mileage')])

print("Predicted Spare Parts:")
for feature, value in predicted_spare_parts.items():
    print(f"{feature}: {value}")

print("Remaining Mileage:")
for feature, mileage in remaining_mileage.items():
    print(f"{feature}: {mileage} miles")

# Step 6: Save the user input and predicted total cost to an output file
output_file = 'predicted_total_cost_and_spare_parts.txt'
with open(output_file, 'w') as f:
    f.write("User Input:\n")
    for feature, value in zip(selected_features, user_input[0]):
        f.write(f"{feature}: {value}\n")
    f.write(
        f"\nPredicted Spare Parts and Cost: {predicted_spare_parts}\n")

print(f"User input and predicted spare parts and cost saved to {output_file}")

# Step 7: Save the model for future use
joblib.dump(model, 'spare_parts_model.joblib')
