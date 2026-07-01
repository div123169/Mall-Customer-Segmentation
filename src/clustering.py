import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import joblib

def run_kmeans_search(X, min_k=2, max_k=10):
    """
    Computes WCSS (inertia) and Silhouette Scores for a range of K values.
    
    Returns:
        k_values: list of range from min_k to max_k
        wcss_list: list of WCSS values corresponding to each K
        silhouette_scores: dict mapping K -> average silhouette score
    """
    k_values = list(range(min_k, max_k + 1))
    wcss_list = []
    silhouette_scores = {}
    
    for k in k_values:
        # Explicitly set n_init=10 to suppress FutureWarning and ensure stability
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss_list.append(kmeans.inertia_)
        
        # Calculate silhouette score (requires at least 2 clusters)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores[k] = score
        
    return k_values, wcss_list, silhouette_scores

def determine_best_k(silhouette_scores):
    """
    Automatically selects the optimal K that maximizes the Silhouette Score.
    """
    if not silhouette_scores:
        return 5  # Safe default for Mall Customer Segmentation
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    return best_k

def train_kmeans(X, n_clusters):
    """
    Trains a K-Means clustering model with the specified number of clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def train_hierarchical(X, n_clusters):
    """
    Trains an Agglomerative Hierarchical clustering model.
    Uses 'metric' parameter with a fallback to 'affinity' for compatibility with older sklearn versions.
    """
    try:
        # scikit-learn >= 1.2
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    except TypeError:
        # scikit-learn < 1.2
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        
    labels = model.fit_predict(X)
    return model, labels

def train_dbscan(X, eps=0.5, min_samples=5):
    """
    Trains a DBSCAN model to detect density-based clusters and potential noise/outliers.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels

def save_model_artifact(model, file_path):
    """
    Saves a trained model or scaler as a binary pickle file using joblib.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Artifact successfully saved to: {file_path}")

def load_model_artifact(file_path):
    """
    Loads a saved joblib artifact.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model artifact not found at: {file_path}")
    return joblib.load(file_path)
