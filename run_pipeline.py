import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_data, save_plot, init_directories, DATASET_PATH, MODELS_DIR, PLOTS_DIR, ASSETS_DIR
from src.preprocessing import preprocess_pipeline
from src.clustering import run_kmeans_search, determine_best_k, train_kmeans, train_hierarchical, train_dbscan, save_model_artifact
from src.evaluate import get_comparison_table

def run_eda(df):
    """
    Step 2: Generate and save EDA plots.
    """
    print("\n=== STEP 2: Running Exploratory Data Analysis ===")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Age Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Age'], kde=True, bins=20, ax=ax, color='skyblue', edgecolor='black')
    ax.set_title('Age Distribution of Mall Customers', fontsize=14, pad=15)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    save_plot(fig, 'age_distribution.png')
    plt.close(fig)
    
    # 2. Gender Distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    gender_counts = df['Gender'].value_counts()
    colors = ['lightcoral', 'cornflowerblue']
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.05, 0))
    ax.set_title('Gender Distribution of Customers', fontsize=14, pad=15)
    save_plot(fig, 'gender_distribution.png')
    plt.close(fig)
    
    # 3. Income Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Annual Income (k$)'], kde=True, bins=15, ax=ax, color='salmon', edgecolor='black')
    ax.set_title('Annual Income Distribution', fontsize=14, pad=15)
    ax.set_xlabel('Annual Income (k$)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    save_plot(fig, 'income_distribution.png')
    plt.close(fig)
    
    # 4. Spending Score Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Spending Score (1-100)'], kde=True, bins=15, ax=ax, color='lightgreen', edgecolor='black')
    ax.set_title('Spending Score Distribution', fontsize=14, pad=15)
    ax.set_xlabel('Spending Score (1-100)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    save_plot(fig, 'spending_score_distribution.png')
    plt.close(fig)
    
    # Pre-encode Gender temporarily for correlation heatmap
    temp_df = df.copy()
    temp_df['Gender_Encoded'] = temp_df['Gender'].map({'Female': 0, 'Male': 1})
    num_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Encoded']
    
    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = temp_df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Heatmap', fontsize=14, pad=15)
    save_plot(fig, 'correlation_heatmap.png')
    plt.close(fig)
    
    # 6. Pairplots
    # Pairplots are handled directly by Seaborn pairplot object which creates its own figure
    pairplot_fig = sns.pairplot(df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']], hue='Gender', palette=colors)
    pairplot_fig.fig.suptitle('Pairplot of Customer Features', y=1.02, fontsize=16)
    save_plot(pairplot_fig.fig, 'pairplot.png')
    plt.close('all')
    
    # 7. Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(x='Gender', y='Age', data=df, ax=axes[0], palette=colors)
    axes[0].set_title('Age by Gender', fontsize=12)
    
    sns.boxplot(x='Gender', y='Annual Income (k$)', data=df, ax=axes[1], palette=colors)
    axes[1].set_title('Annual Income by Gender', fontsize=12)
    
    sns.boxplot(x='Gender', y='Spending Score (1-100)', data=df, ax=axes[2], palette=colors)
    axes[2].set_title('Spending Score by Gender', fontsize=12)
    
    fig.suptitle('Feature Boxplots by Gender', fontsize=16, y=1.02)
    save_plot(fig, 'boxplots_by_gender.png')
    plt.close(fig)
    
    # 8. Scatter Plot (Income vs Spending Score)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', size='Age', 
                    sizes=(20, 200), palette=colors, alpha=0.8, data=df, ax=ax)
    ax.set_title('Annual Income vs Spending Score (Colored by Gender, Size by Age)', fontsize=14, pad=15)
    ax.set_xlabel('Annual Income (k$)', fontsize=12)
    ax.set_ylabel('Spending Score (1-100)', fontsize=12)
    save_plot(fig, 'income_vs_spend_scatter.png')
    plt.close(fig)
    print("EDA Visualizations saved successfully.")

def run_pipeline():
    # Ensure folders are initialized
    init_directories()
    
    # STEP 1: Load the real dataset and perform Overview Analysis
    print("=== STEP 1: Loading Dataset and Performing Overview ===")
    df = load_data()
    
    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Missing Value Analysis
    missing_vals = df.isnull().sum()
    print("\nMissing Value Analysis:")
    print(missing_vals)
    
    # Duplicate Analysis
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate Rows count: {dup_count}")
    
    # Statistical Summary
    print("\nStatistical Summary:")
    print(df.describe())
    
    # STEP 2: EDA
    run_eda(df)
    
    # STEP 3: Preprocessing
    print("\n=== STEP 3: Preprocessing Data ===")
    features_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    df_preprocessed, scaler = preprocess_pipeline(df, features_to_scale=features_to_scale)
    
    # Get scaled numerical features for clustering
    # We cluster on Age, Annual Income, Spending Score
    X = df_preprocessed[features_to_scale].values
    
    # STEP 4: Train clustering models & Elbow / Silhouette search
    print("\n=== STEP 4: Hyperparameter Tuning and Clustering ===")
    k_values, wcss_list, silhouette_scores = run_kmeans_search(X, min_k=2, max_k=10)
    
    best_k = determine_best_k(silhouette_scores)
    print(f"Optimal number of clusters (K) automatically selected: {best_k}")
    
    # Train Models
    print(f"Training final K-Means with K={best_k}...")
    kmeans_model, kmeans_labels = train_kmeans(X, best_k)
    
    print(f"Training final Hierarchical Agglomerative with K={best_k}...")
    hier_model, hier_labels = train_hierarchical(X, best_k)
    
    print("Training DBSCAN model (eps=0.5, min_samples=5)...")
    dbscan_model, dbscan_labels = train_dbscan(X, eps=0.5, min_samples=5)
    
    # STEP 5: Generate Visualizations for Clustering
    print("\n=== STEP 5: Generating Clustering Visualizations ===")
    
    # 1. Elbow Curve Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, wcss_list, marker='o', linestyle='-', color='indigo')
    ax.set_title('K-Means Elbow Method for Optimal K', fontsize=14, pad=15)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('WCSS (Inertia)', fontsize=12)
    ax.set_xticks(k_values)
    save_plot(fig, 'kmeans_elbow_curve.png')
    plt.close(fig)
    
    # 2. Silhouette Score Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    scores_list = [silhouette_scores[k] for k in k_values]
    ax.plot(k_values, scores_list, marker='s', linestyle='--', color='teal')
    ax.axvline(x=best_k, color='red', linestyle=':', label=f'Optimal K ({best_k})')
    ax.set_title('K-Means Silhouette Scores', fontsize=14, pad=15)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Average Silhouette Score', fontsize=12)
    ax.set_xticks(k_values)
    ax.legend()
    save_plot(fig, 'kmeans_silhouette_scores.png')
    plt.close(fig)
    
    # 3. 2D Cluster Plot (Annual Income vs Spending Score)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], 
                    hue=kmeans_labels, palette='viridis', style=kmeans_labels, s=100, alpha=0.8, ax=ax)
    
    # Get centroids in original scale for plotting
    centroids_original = scaler.inverse_transform(kmeans_model.cluster_centers_)
    # Features order is Age, Income, Spend
    income_idx = features_to_scale.index('Annual Income (k$)')
    spend_idx = features_to_scale.index('Spending Score (1-100)')
    
    ax.scatter(centroids_original[:, income_idx], centroids_original[:, spend_idx], 
               s=300, c='red', marker='X', edgecolors='black', label='Centroids')
    ax.set_title(f'K-Means Clusters (K={best_k})', fontsize=14, pad=15)
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.legend(title='Clusters')
    save_plot(fig, 'kmeans_clusters_2d.png')
    plt.close(fig)
    
    # 4. PCA Cluster Visualization
    # Apply PCA to project 3D scaled features down to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='tab10', s=80, alpha=0.8, ax=ax)
    ax.set_title('K-Means Clusters Project on PCA 2D Space', fontsize=14, pad=15)
    ax.set_xlabel('PCA Principal Component 1')
    ax.set_ylabel('PCA Principal Component 2')
    ax.legend(title='Clusters')
    save_plot(fig, 'kmeans_clusters_pca.png')
    plt.close(fig)
    
    # STEP 6: Save the best clustering model and scaler
    print("\n=== STEP 6: Saving Best Clustering Model ===")
    save_model_artifact(kmeans_model, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    save_model_artifact(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    
    # Compile comparison table
    print("\n=== MODEL COMPARISON TABLE ===")
    comparison_table = get_comparison_table(X, kmeans_labels, hier_labels, dbscan_labels)
    print(comparison_table)
    
    # Save comparison table to csv
    comparison_table.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"))
    print("\nPipeline completed successfully! All models and visual assets generated.")

if __name__ == "__main__":
    run_pipeline()
