import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

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




