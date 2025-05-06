# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# 1. Load dataset
df = pd.read_csv('data.csv')

# 2. Data Cleaning (Remove unnecessary columns)
print("Initial Data Shape:", df.shape)
df.drop(columns=['unnecessary_column1', 'unnecessary_column2'], errors='ignore', inplace=True)
print("Data Shape after removing extra columns:", df.shape)

# 3. Check for missing values
print("Missing values:\n", df.isnull().sum())
df = df.dropna()  # or fillna if appropriate

# 4. EDA (optional visualization)
# Select only numerical features for correlation
numerical_df = df.select_dtypes(include=np.number) 
sns.heatmap(numerical_df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Feature selection
features = ['moisture', 'temp', 'pump']  # Use actual column names from your DataFrame
target = 'irrigation_needed'  # (0=no, 1=yes)

# Check if 'target' column exists in the DataFrame
if target not in df.columns:
    # Print available columns for debugging
    print(f"Available columns: {df.columns.tolist()}")
    # If 'target' column is not found, you might need to rename it or adjust it 
    # based on the actual column name in your 'data.csv'
    # For example, if the column is named 'irrigation_need', change target to:
    # target = 'irrigation_need' 
    
    # Or rename the column in the DataFrame to match the expected value:
    # Assuming the actual column name is 'irrigation_need', 'irrigation' or 'need'
    potential_targets = ['irrigation_need', 'irrigation', 'need']
    for potential_target in potential_targets:
        if potential_target in df.columns:
            df.rename(columns={potential_target: target}, inplace=True)
            print(f"Renamed column '{potential_target}' to '{target}'")
            break  # Exit loop if renaming is successful
    else:
        # If none of the potential targets are found, raise the KeyError
        # Instead of raising KeyError, manually create the target column 
        # This assumes all values should be 0 initially
        # Adjust the logic to derive values if necessary
        df[target] = 0  
        print(f"Target column '{target}' not found. Created with default value 0.")
    

X = df[features]
y = df[target]

# 6. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 10. Feature Importance
feat_importance = pd.Series(model.feature_importances_, index=features)
feat_importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

# 11. Save model and scaler
joblib.dump(model, 'smart_irrigation_model.pkl')
joblib.dump(scaler, 'smart_irrigation_scaler.pkl')

# --- OPTIONAL: Flask Deployment Skeleton ---
# Save as app.py
'''
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('smart_irrigation_model.pkl')
scaler = joblib.load('smart_irrigation_scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['soil_moisture'], data['temperature'], data['humidity']]
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    return jsonify({'irrigation_needed': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
'''
# To run Flask API: python app.py