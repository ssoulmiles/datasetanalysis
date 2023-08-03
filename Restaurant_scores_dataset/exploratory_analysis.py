import pandas as pd

# Load the data
data = pd.read_csv('Restaurant_Scores.csv')

# Performing Exploratory Data Analysis

# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Data types and non-null counts
print(data.info())