# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Import Linear Regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score # Import relevant metrics
import joblib
from sklearn.cluster import KMeans # Import KMeans

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
target = 'irrigation_needed'  # Replace with the actual column name from your CSV 

# Check if 'target' column exists in the DataFrame
if target not in df.columns:
    # If 'target' column doesn't exist, create it based on a condition
    # Replace this condition with your actual logic for determining irrigation need
    df[target] = 0  # Initialize with 0
    df.loc[(df['moisture'] < 30) & (df['temp'] > 25), target] = 1 
    print(f"Target column '{target}' not found in the DataFrame. Created based on a condition.")
    # If you have a different way to derive 'irrigation_needed', implement it here.
    

X = df[features]
y = df[target]

# 6. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Apply KMeans Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42) # Adjust n_clusters as needed
df['cluster'] = kmeans.fit_predict(X_scaled)

# --- Visualize Clusters (Optional) ---
sns.scatterplot(x='moisture', y='temp', hue='cluster', data=df)
plt.title("KMeans Clusters")
plt.show()

# 7. Train-test split (using cluster as a feature)
features.append('cluster')  # Add cluster to features
X = df[features]
X_scaled = scaler.fit_transform(X) # Re-scale after adding cluster
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Model training (Linear Regression)
model = LinearRegression()  
model.fit(X_train, y_train)

# 9. Evaluation (using metrics suitable for regression)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# 10. Feature Importance (Coefficients for Linear Regression)
coefficients = model.coef_
print("Coefficients:\n", coefficients)

# --- Visualizations using Matplotlib ---
# 1. Actual vs. Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Diagonal line
plt.xlabel("Actual Irrigation Needed")
plt.ylabel("Predicted Irrigation Needed")
plt.title("Actual vs. Predicted Values (Linear Regression)")
plt.show()

# 2. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='green', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')  # Horizontal line at 0
plt.xlabel("Predicted Irrigation Needed")
plt.ylabel("Residuals")
plt.title("Residual Plot (Linear Regression)")
plt.show()

# 11. Save model and scaler
joblib.dump(model, 'irrigation_model.pkl')
joblib.dump(scaler, 'irrigation_scaler.pkl')