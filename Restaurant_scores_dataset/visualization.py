import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('Restaurant_Scores.csv')

# Pairplot to visualize relationships between features
sns.pairplot(data, hue='inspection_score', diag_kind='kde')
plt.title('Pairplot of Features with Inspection Score')
plt.show()

# Heatmap of correlation matrix
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Bar chart of target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='inspection_score', data=data)
plt.title('Inspection Score Distribution')
plt.xlabel('Inspection Score')
plt.ylabel('Count')
plt.show()

# Box plots to visualize feature distributions by target class
plt.figure(figsize=(10, 6))
sns.boxplot(x='inspection_score', y='Current Police Districts', data=data)
plt.title('Box Plot of Current Police Districts by Inspection Score')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='inspection_score', y='Neighborhoods', data=data)
plt.title('Box Plot of Neighborhoods by Inspection Score')
plt.show()
