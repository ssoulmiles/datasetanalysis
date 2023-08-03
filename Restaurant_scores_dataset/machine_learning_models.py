import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('Restaurant_Scores.csv')

# Handling Missing Values
data.dropna(inplace=True)  # Remove rows with missing values

# Drop non-numerical values
data = data.drop(['business_id', 'business_name', 'business_address', 'business_city', 'business_state', 
                  'business_postal_code', 'business_location', 'inspection_id', 'inspection_date', 'inspection_type', 
                  'violation_id', 'violation_description', 'risk_category'], axis='columns')
 
# Print the dataset
print(data.head())

# Review data types
print(data.dtypes)
print("\n")


# IQR (Interquartile Range) Method to determine outliers
Q1 = data['inspection_score'].quantile(0.25)
Q3 = data['inspection_score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['inspection_score'] < lower_bound) | (data['inspection_score'] > upper_bound)]

# Remove outliers
data = data[~((data['inspection_score'] < lower_bound) | (data['inspection_score'] > upper_bound))]

print(data.head())

# Set the target column to inspection_score
X = data.drop('inspection_score', axis=1)
y = data['inspection_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
svm_predictions = svm_model.predict(X_test_scaled)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
rf_predictions = rf_model.predict(X_test_scaled)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
