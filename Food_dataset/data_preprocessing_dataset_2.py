import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

# Load the data
data = pd.read_csv('Food_USDA.csv')

# Handling Missing Values
data.dropna(inplace=True)  # Remove rows with missing values

# Review data types to determine non-numeric columns
print(data.dtypes)

# Drop non-numerical values
data = data.drop(['Shrt_Desc', 'GmWt_Desc1', 'GmWt_Desc2'], axis='columns')
 
# Print the dataset
print(data.head())

# IQR (Interquartile Range) Method to determine outliers
Q1 = data['Protein_(g)'].quantile(0.25)
Q3 = data['Protein_(g)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['Protein_(g)'] < lower_bound) | (data['Protein_(g)'] > upper_bound)]

# Remove outliers
data = data[~((data['Protein_(g)'] < lower_bound) | (data['Protein_(g)'] > upper_bound))]

print(data.head())



