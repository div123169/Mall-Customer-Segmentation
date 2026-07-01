import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clustering(X, labels):
    """
    Computes silhouette, Calinski-Harabasz, and Davies-Bouldin validation metrics.
    Robust against edge cases, such as when DBSCAN produces only noise or single clusters.
    
    Returns:
        metrics_dict: dict of computed clustering metrics
    """
    unique_labels = set(labels)
    # Exclude noise label -1 (for DBSCAN) in cluster counts
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # We need at least 2 clusters and less than N clusters to compute metrics
    if n_clusters < 2:
        return {
            "Silhouette Score": -1.0,
            "Calinski-Harabasz Index": 0.0,
            "Davies-Bouldin Index": 999.0,
            "Number of Clusters": int(n_clusters),
            "Note": "Evaluation skipped: less than 2 non-noise clusters formed."
        }
    
    # Filter out noise points if analyzing DBSCAN to get pure cluster statistics
    # (Optional, but standard sklearn metrics fail if only noise and 1 cluster exist)
    mask = (labels != -1)
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    if len(set(labels_clean)) < 2:
        return {
            "Silhouette Score": -1.0,
            "Calinski-Harabasz Index": 0.0,
            "Davies-Bouldin Index": 999.0,
            "Number of Clusters": int(n_clusters),
            "Note": "Evaluation skipped: less than 2 clusters remaining after filtering noise."
        }
        
    sil = silhouette_score(X_clean, labels_clean)
    ch = calinski_harabasz_score(X_clean, labels_clean)
    db = davies_bouldin_score(X_clean, labels_clean)
    
    return {
        "Silhouette Score": round(float(sil), 4),
        "Calinski-Harabasz Index": round(float(ch), 2),
        "Davies-Bouldin Index": round(float(db), 4),
        "Number of Clusters": int(n_clusters)
    }

def get_comparison_table(X, kmeans_labels, hierarchical_labels, dbscan_labels):
    """
    Assembles a comparison DataFrame of metrics for the three algorithms.
    """
    kmeans_eval = evaluate_clustering(X, kmeans_labels)
    hier_eval = evaluate_clustering(X, hierarchical_labels)
    dbscan_eval = evaluate_clustering(X, dbscan_labels)
    
    comparison_data = {
        "K-Means": kmeans_eval,
        "Hierarchical (Agglomerative)": hier_eval,
        "DBSCAN": dbscan_eval
    }
    
    df = pd.DataFrame(comparison_data).T
    # Order columns nicely
    cols = ["Number of Clusters", "Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]
    # Check if they exist in output
    actual_cols = [c for c in cols if c in df.columns]
    if "Note" in df.columns:
        actual_cols.append("Note")
    return df[actual_cols]
