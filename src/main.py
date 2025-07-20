# -*- coding: utf-8 -*-


import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
from IPython.display import display
os.makedirs("output", exist_ok=True)
plot_count = 0
def savefig(title):
    global plot_count
    plot_count += 1
    filename = f"output/plot_{plot_count}_{title.replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(filename)
    plt.close()


file_path = "data/tomato_irrigation_dataset.csv"
df = pd.read_csv(file_path)

print("\nDataset Info:")
df.info()

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nSummary statistics:")
print(df.describe())

print("\nColumn names and their data types:")
print(df.dtypes)

print("\nNumber of rows and columns:")
print(df.shape)

"""Code for find the missing value"""

missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

numerical_cols = df.select_dtypes(include=np.number).columns

#Select numerical columns and create box plots to visualize potential outliers.
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i + 1)
    df.boxplot(column=col)
    plt.title(col)

plt.tight_layout()
savefig("figure_1")

#Calculate Z-scores and identify outliers using a threshold.
outlier_indices_zscore = {}
for col in numerical_cols:
    z_scores = np.abs(zscore(df[col]))
    outlier_indices = df[z_scores > 3].index
    outlier_indices_zscore[col] = outlier_indices

print("Potential outliers identified by Z-score (threshold=3):")
for col, indices in outlier_indices_zscore.items():
    if len(indices) > 0:
        print(f"Column '{col}': {list(indices)}")

"""outlier handling Using Cap method"""

outlier_strategy = {
    'pH': 'cap'
}

for col, strategy in outlier_strategy.items():
    if col in outlier_indices_zscore and len(outlier_indices_zscore[col]) > 0:
       if strategy == 'cap':
            upper_limit = df[col].mean() + 3 * df[col].std()
            lower_limit = df[col].mean() - 3 * df[col].std()
            df[col] = np.where(df[col] > upper_limit, upper_limit,
                               np.where(df[col] < lower_limit, lower_limit, df[col]))
            print(f"Capped outliers in column '{col}'.")
display(df.head())

"""Check if the outliers have been effectively handled"""

numerical_cols = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i + 1)
    df.boxplot(column=col)
    plt.title(col)

plt.tight_layout()
savefig("figure_2")

from scipy.stats import zscore

z_scores = np.abs(zscore(df[numerical_cols]))
outlier_indices_zscore = {}
for col in numerical_cols:
    outlier_indices = df[np.abs(zscore(df[col])) > 3].index
    outlier_indices_zscore[col] = outlier_indices

print("Potential outliers identified by Z-score (threshold=3) after handling:")
for col, indices in outlier_indices_zscore.items():
    if len(indices) > 0:
        print(f"Column '{col}': {list(indices)}")

"""Generate Scatter plots to visualize the relationship between each nutrient level ('Nitrogen [mg/kg]', 'Phosphorus [mg/kg]', and 'Potassium') and 'Days of planted"""

nutrient_cols = ['Nitrogen [mg/kg]', 'Phosphorus [mg/kg]', 'Potassium']

plt.figure(figsize=(15, 5))

for i, col in enumerate(nutrient_cols):
    plt.subplot(1, 3, i + 1)
    plt.scatter(df['Days of planted'], df[col]) # Changed to plt.scatter for scatter plot
    plt.xlabel('Days of planted')
    plt.ylabel(col)
    plt.title(f'{col} vs. Days of planted')
    plt.grid(True)

plt.tight_layout()
savefig("figure_3")

"""Use the Elbow Method Nutrient clusters"""

nutrient_cols = ['Nitrogen [mg/kg]', 'Phosphorus [mg/kg]', 'Potassium']
max_clusters = 10

for col in nutrient_cols:
    X = df[['Days of planted', col]].dropna().values

    if len(X) > 0:
        inertia = []
        for n_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_clusters + 1), inertia, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title(f'Elbow Method for {col}')
        plt.grid(True)
        savefig("figure_4")
    else:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No data to cluster", horizontalalignment='center', verticalalignment='center')
        plt.title(f'Elbow Method for {col}')
        savefig("figure_5")

"""Apply K-means clustering to each nutrient level vs. 'Days of planted' plot and visualize the clusters."""

nutrient_cols = ['Nitrogen [mg/kg]', 'Phosphorus [mg/kg]', 'Potassium']
n_clusters = {
    'Nitrogen [mg/kg]': 3,
    'Phosphorus [mg/kg]': 3,
    'Potassium': 3
}


for i, col in enumerate(nutrient_cols):
    X = df[['Days of planted', col]].dropna().values

    if len(X) > 0:
        kmeans = KMeans(n_clusters=n_clusters[col], random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        # Add cluster labels to the DataFrame
        cluster_col_name = f'{col}_DaysPlanted_KMeans_Cluster'
        # Align the cluster labels with the original DataFrame indices
        df.loc[df[['Days of planted', col]].dropna().index, cluster_col_name] = clusters


        plt.figure(figsize=(8, 5))
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
        plt.xlabel('Days of planted')
        plt.ylabel(col)
        plt.title(f'{col} vs. Days of planted (K-Means Clustering with {n_clusters[col]} Clusters)')
        plt.grid(True)
        savefig("figure_6")
    else:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No data to cluster", horizontalalignment='center', verticalalignment='center')
        plt.title(f'{col} vs. Days of planted (K-Means Clustering)')
        savefig("figure_7")

"""clustering evaluation metrics and calculate them for each nutrient based on the previously generated K-means clusters."""

nutrient_cols = ['Nitrogen [mg/kg]', 'Phosphorus [mg/kg]', 'Potassium']

for col in nutrient_cols:
    cluster_col = f'{col}_DaysPlanted_KMeans_Cluster'


    if cluster_col in df.columns:

        X = df[['Days of planted', col]].dropna().values

        cluster_labels = df[cluster_col].dropna()


        if len(X) > 1 and len(set(cluster_labels)) > 1:
            print(f"Evaluating cluster quality for '{col}' vs. 'Days of planted':")

            try:
                silhouette_avg = silhouette_score(X, cluster_labels)


                print(f"  Silhouette Score: {silhouette_avg:.4f}")

            except ValueError as e:
                print(f"  Could not calculate metrics: {e}")
        elif len(X) <= 1:
            print(f"Not enough data points to calculate metrics for '{col}' vs. 'Days of planted'.")
        else:
             print(f"Only one cluster found for '{col}' vs. 'Days of planted', metrics cannot be calculated.")
    else:
        print(f"Cluster column '{cluster_col}' not found in the DataFrame.")

    print("-" * 50)

"""visualize the clusters and their distribution"""

import matplotlib.pyplot as plt

nutrient_cols = ['Nitrogen [mg/kg]', 'Phosphorus [mg/kg]', 'Potassium']

plt.figure(figsize=(18, 6))

for i, col in enumerate(nutrient_cols):
    cluster_col = f'{col}_DaysPlanted_KMeans_Cluster'
    if cluster_col in df.columns:
        plt.subplot(1, 3, i + 1)
        plt.scatter(df['Days of planted'], df[col], c=df[cluster_col], cmap='viridis')
        plt.xlabel('Days of planted')
        plt.ylabel(col)
        plt.title(f'K-Means Clusters of {col} vs. Days of planted')
        plt.colorbar(label='Cluster')
        plt.grid(True)
    else:
        print(f"Cluster column '{cluster_col}' not found in the DataFrame.")

plt.tight_layout()
savefig("figure_8")

"""Analyze characteristics of each cluster using descriptive statistics"""

nutrient_cols = ['Nitrogen [mg/kg]', 'Phosphorus [mg/kg]', 'Potassium']

for col in nutrient_cols:
    cluster_col = f'{col}_DaysPlanted_KMeans_Cluster'
    if cluster_col in df.columns:
        print(f"Descriptive statistics for clusters of '{col}' vs. 'Days of planted':")


        grouped_stats = df.groupby(cluster_col)[['Days of planted', col]].agg(['mean', 'median', 'std'])

        display(grouped_stats)
        print("-" * 50)
    else:
        print(f"Cluster column '{cluster_col}' not found in the DataFrame.")
        print("-" * 50)

"""Create box plots to visualize the distribution of 'Days of planted' and nutrient levels across the clusters"""

nutrient_cols = ['Nitrogen [mg/kg]', 'Phosphorus [mg/kg]', 'Potassium']

for col in nutrient_cols:
    cluster_col = f'{col}_DaysPlanted_KMeans_Cluster'
    if cluster_col in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        df.boxplot(column='Days of planted', by=cluster_col, ax=axes[0])
        axes[0].set_title(f'Days of planted distribution by {col} Cluster')
        axes[0].set_xlabel(f'{col} Cluster')
        axes[0].set_ylabel('Days of planted')

        df.boxplot(column=col, by=cluster_col, ax=axes[1])
        axes[1].set_title(f'{col} distribution by {col} Cluster')
        axes[1].set_xlabel(f'{col} Cluster')
        axes[1].set_ylabel(col)

        plt.suptitle(f'Cluster Distributions for {col}', y=1.02)
        plt.tight_layout()
        savefig("figure_9")
    else:
        print(f"Cluster column '{cluster_col}' not found in the DataFrame.")

"""Analyzing trends"""

nutrient_cols = ['Nitrogen [mg/kg]', 'Phosphorus [mg/kg]', 'Potassium']

for col in nutrient_cols:
    cluster_col = f'{col}_DaysPlanted_KMeans_Cluster'
    if cluster_col in df.columns:
        print(f"Analyzing trends for '{col}' across clusters:")

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Days of planted', y=col, hue=cluster_col, palette='viridis')
        plt.title(f'{col} Trend over Days of planted by Cluster')
        plt.xlabel('Days of planted')
        plt.ylabel(col)
        plt.grid(True)
        savefig("figure_10")
    else:
        print(f"Cluster column '{cluster_col}' not found in the DataFrame.")
    print("-" * 50)

"""Analyzing 'Crop Coefficient stage' distribution within nutrients' clusters"""

nutrient_cols = ['Nitrogen [mg/kg]', 'Phosphorus [mg/kg]', 'Potassium']
crop_stage_col = 'Crop Coefficient stage'

if crop_stage_col in df.columns:
    fig, axes = plt.subplots(1, len(nutrient_cols), figsize=(18, 6)) # Create a figure with subplots in one row

    for i, col in enumerate(nutrient_cols):
        nutrient_cluster_col = f'{col}_DaysPlanted_KMeans_Cluster'
        if nutrient_cluster_col in df.columns:
            print(f"Analyzing '{crop_stage_col}' distribution within '{col}' clusters:")

            sns.countplot(data=df, x=nutrient_cluster_col, hue=crop_stage_col, palette='viridis', ax=axes[i]) # Plot on the i-th subplot
            axes[i].set_title(f'Distribution of Crop Coefficient Stage within {col} Clusters')
            axes[i].set_xlabel(f'{col} Cluster')
            axes[i].set_ylabel('Count')
            axes[i].grid(axis='y')
            axes[i].legend(title='Crop Stage') # Add legend to each subplot
        else:
            print(f"Nutrient cluster column '{nutrient_cluster_col}' not found in the DataFrame.")
        print("-" * 50)

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    savefig("figure_11")

else:
    print(f"Required column '{crop_stage_col}' not found in the DataFrame.")

"""pH vs. Days of planted scatter plot"""

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Days of planted', y='pH')
plt.title('pH vs. Days of planted')
plt.xlabel('Days of planted')
plt.ylabel('pH')
plt.grid(True)
savefig("figure_12")

"""Apply L Bow method"""

X = df[['Days of planted', 'pH']].dropna().values
max_clusters = 10
inertia = []

if len(X) > 0:
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for pH vs. Days of planted')
    plt.grid(True)
    savefig("figure_13")
else:
    plt.figure(figsize=(8, 5))
    plt.text(0.5, 0.5, "No data to cluster", horizontalalignment='center', verticalalignment='center')
    plt.title('Elbow Method for pH vs. Days of planted')
    savefig("figure_14")

"""Apply K Menas Clustering"""

X = df[['Days of planted', 'pH']].dropna().values
n_clusters_ph = 3

if len(X) > 0:
    kmeans_ph = KMeans(n_clusters=n_clusters_ph, random_state=42, n_init=10)
    clusters_ph = kmeans_ph.fit_predict(X)

    cluster_col_name_ph = 'pH_DaysPlanted_KMeans_Cluster'

    temp_df = df[['Days of planted', 'pH']].dropna()

    temp_df[cluster_col_name_ph] = clusters_ph

    df = df.merge(temp_df[[cluster_col_name_ph]], left_index=True, right_index=True, how='left')


    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=clusters_ph, cmap='viridis')
    plt.xlabel('Days of planted')
    plt.ylabel('pH')
    plt.title(f'pH vs. Days of planted (K-Means Clustering with {n_clusters_ph} Clusters)')
    plt.grid(True)
    savefig("figure_15")
else:
    plt.figure(figsize=(8, 5))
    plt.text(0.5, 0.5, "No data to cluster", horizontalalignment='center', verticalalignment='center')
    plt.title('pH vs. Days of planted (K-Means Clustering)')
    savefig("figure_16")

"""pH_DaysPlanted_KMeans_Cluster with naming"""

cluster_col_name_ph = 'pH_DaysPlanted_KMeans_Cluster'

if cluster_col_name_ph in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='Days of planted', y='pH', hue=cluster_col_name_ph, palette='viridis')
    plt.title(f'pH vs. Days of planted (K-Means Clustering)')
    plt.xlabel('Days of planted')
    plt.ylabel('pH')
    plt.grid(True)
    savefig("figure_17")
else:
    print(f"Cluster column '{cluster_col_name_ph}' not found in the DataFrame.")

"""Descriptive statistics for clusters of 'pH' vs. 'Days of planted'"""

cluster_col_name_ph = 'pH_DaysPlanted_KMeans_Cluster'

if cluster_col_name_ph in df.columns:
    print(f"Descriptive statistics for clusters of 'pH' vs. 'Days of planted':")

    grouped_stats_ph = df.groupby(cluster_col_name_ph)[['Days of planted', 'pH']].agg(['mean', 'median', 'std'])

    display(grouped_stats_ph)
else:
    print(f"Cluster column '{cluster_col_name_ph}' not found in the DataFrame.")

cluster_col_name_ph = 'pH_DaysPlanted_KMeans_Cluster'

if cluster_col_name_ph in df.columns:
    X = df[['Days of planted', 'pH']].dropna().values
    cluster_labels_ph = df[cluster_col_name_ph].dropna()

    if len(np.unique(cluster_labels_ph)) > 1 and len(X) > 1:
        silhouette_avg_ph = silhouette_score(X, cluster_labels_ph)
        print(f"Silhouette Score for pH vs. Days of planted clusters: {silhouette_avg_ph:.4f}")
    elif len(X) <= 1:
        print("Not enough data points to calculate Silhouette score.")
    else:
        print("Only one cluster found, cannot calculate Silhouette score.")
else:
    print(f"Cluster column '{cluster_col_name_ph}' not found in the DataFrame.")

"""Distribution of Crop Coefficient Stage within pH Clusters"""

cluster_col_name_ph = 'pH_DaysPlanted_KMeans_Cluster'

if cluster_col_name_ph in df.columns and 'Crop Coefficient stage' in df.columns:
    print("Analyzing 'Crop Coefficient stage' distribution within pH clusters:")

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=cluster_col_name_ph, hue='Crop Coefficient stage', palette='viridis')
    plt.title('Distribution of Crop Coefficient Stage within pH Clusters')
    plt.xlabel('pH Cluster')
    plt.ylabel('Count')
    plt.grid(axis='y')
    savefig("figure_18")

else:
    print(f"Required columns ('{cluster_col_name_ph}' or 'Crop Coefficient stage') not found in the DataFrame.")

print("-" * 50)

ph_cluster_col = 'pH_DaysPlanted_KMeans_Cluster'

if ph_cluster_col in df.columns:
    print(f"Analyzing trends for 'pH' across clusters:")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Days of planted', y='pH', hue=ph_cluster_col, palette='viridis')
    plt.title(f'pH over Days of planted by Cluster')
    plt.xlabel('Days of planted')
    plt.ylabel('pH')
    plt.grid(True)
    savefig("figure_19")
else:
    print(f"Cluster column '{ph_cluster_col}' not found in the DataFrame.")

print("-" * 50)

nutrient_cols = ['Nitrogen [mg/kg]', 'Phosphorus [mg/kg]', 'Potassium']
ph_cluster_col = 'pH_DaysPlanted_KMeans_Cluster'

if ph_cluster_col in df.columns:
    fig, axes = plt.subplots(1, len(nutrient_cols), figsize=(18, 6))

    for i, col in enumerate(nutrient_cols):
        nutrient_cluster_col = f'{col}_DaysPlanted_KMeans_Cluster'
        if nutrient_cluster_col in df.columns:
            print(f"Analyzing '{ph_cluster_col}' distribution within '{col}' clusters:")

            sns.countplot(data=df, x=nutrient_cluster_col, hue=ph_cluster_col, palette='viridis', ax=axes[i])
            axes[i].set_title(f'Distribution of pH Clusters within {col} Clusters')
            axes[i].set_xlabel(f'{col} Cluster')
            axes[i].set_ylabel('Count')
            axes[i].grid(axis='y')
            axes[i].legend(title='pH Cluster')
        else:
            print(f"Nutrient cluster column '{nutrient_cluster_col}' not found in the DataFrame.")
        print("-" * 50)

    plt.tight_layout()
    savefig("figure_20")

else:
    print(f"pH cluster column '{ph_cluster_col}' not found in the DataFrame.")