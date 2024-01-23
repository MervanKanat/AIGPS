"""
Created on Fri Jun 30 19:15:21 2023

@author: bilgi
"""
# Necessary libraries and modules are imported.
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.patches as mpatches

# Ignoring all warnings
warnings.filterwarnings("ignore")

# Load the dataset from an Excel file
df = pd.read_excel('GpsDataSet.xlsx')

# Initialize a dictionary to store the counts of outliers for each n_neighbors value
outlier_counts = {}

# Loop over the n_neighbors parameter from 1 to 10 to identify optimal number of neighbors for Local Outlier Factor model
for n in range(1, 11):
    # Create and train the LOF model on Latitude column
    clf = LocalOutlierFactor(n_neighbors=n)
    pred = clf.fit_predict(df[['Latitude']])

    # Identify the outliers (-1 indicates outliers)
    outliers = df[pred == -1]

    # Add the count of outliers to the dictionary
    outlier_counts[n] = len(outliers)

# Convert the dictionary results to a DataFrame for easy manipulation and plotting
outlier_counts_df = pd.DataFrame(list(outlier_counts.items()), columns=['n_neighbors', 'Outlier Count'])

# Print the DataFrame of outlier counts
print(outlier_counts_df)

# Plotting setup
fig, axs = plt.subplots(1, 2, figsize=(17, 7))  # 1 row, 2 columns

# Identify rows with the maximum outlier count
max_outlier_rows = outlier_counts_df[outlier_counts_df['Outlier Count'] == outlier_counts_df['Outlier Count'].max()]

# Print the rows with the maximum outlier count in red
print(f"\033[31mThe rows with the highest number of outliers are:\n{max_outlier_rows}\033[0m")

# Create a color list for scatter plot points, outliers are marked in red, inliers in blue
colors = ['red' if i in max_outlier_rows.index else 'blue' for i in df.index]

# Scatter plot of Latitude values, outliers are marked in red
scatter = axs[0].scatter(df.index, df['Latitude'], c=colors)
axs[0].set_title('Outliers in Latitude')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Latitude')
axs[0].grid(True)

# Create a legend for the scatter plot
red_patch = mpatches.Patch(color='red', label='Outlier')
blue_patch = mpatches.Patch(color='blue', label='Normal')
axs[0].legend(handles=[red_patch, blue_patch])

# Line plot showing Outlier Count vs n_neighbors
line = axs[1].plot(outlier_counts_df['n_neighbors'], outlier_counts_df['Outlier Count'], marker='o')
axs[1].set_title('Outlier Count vs n_neighbors')
axs[1].set_xlabel('n_neighbors')
axs[1].set_ylabel('Outlier Count')
axs[1].grid(True)


from scipy.stats import zscore

def detect_outliers_with_zscore(data, threshold=3):
    # Compute Z-Scores
    z_scores = zscore(data)
    # Identify Outliers
    outliers = data[(z_scores > threshold) | (z_scores < -threshold)]
    return outliers

for column in df.columns:
    outliers = detect_outliers_with_zscore(df[column])
    if not outliers.empty:
        print(f"Outliers detected by Z-Score method in '{column}' column:\n", outliers, "\n")
 


# Show the plots
plt.show()
