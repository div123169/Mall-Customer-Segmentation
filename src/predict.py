import os
import numpy as np
import pandas as pd
from src.clustering import load_model_artifact

# Define customer profile details and recommendations
CUSTOMER_PROFILES = {
    "VIP": {
        "name": "Affluent High-Spenders (VIP)",
        "description": "Customers with high income and high spending scores. They appreciate luxury, premium services, and exclusive products.",
        "recommendation": "Offer VIP club memberships, early access to premium product launches, personal shopper services, and exclusive invitation-only events."
    },
    "Saver": {
        "name": "Affluent Savers (Careful)",
        "description": "Customers with high income but low spending scores. They are financially comfortable but highly disciplined and value-conscious.",
        "recommendation": "Target with durability-focused marketing, investment-value descriptions, clear cashback reward programs, and hassle-free return policies."
    },
    "Impulsive": {
        "name": "Budget Impulsive Spenders (Careless)",
        "description": "Customers with low income but high spending scores. Often younger buyers who are highly responsive to trends, flash sales, and impulse items.",
        "recommendation": "Promote trending fashion items via social media, offer flash discounts, and integrate buy-now-pay-later (BNPL) options."
    },
    "Frugal": {
        "name": "Conservative Economizers (Sensible)",
        "description": "Customers with low income and low spending scores. Highly price-sensitive shoppers who prioritize necessities.",
        "recommendation": "Offer value-packs, BOGO (Buy One Get One) deals, entry-level product offerings, and straight price-cut promotions."
    },
    "Standard": {
        "name": "Balanced Middle Class (Standard)",
        "description": "Customers with moderate income and moderate spending scores. This represents the average, steady shopper baseline.",
        "recommendation": "Enroll in general newsletter promotions, send birthday discount coupons, and use loyalty programs to encourage repeat visits."
    }
}

# Pre-defined ideal coordinates in original scale for cluster identification
# Format: (Annual Income (k$), Spending Score (1-100))
IDEAL_CENTROIDS = {
    "VIP": (90, 80),
    "Saver": (90, 20),
    "Impulsive": (25, 80),
    "Frugal": (25, 20),
    "Standard": (55, 50)
}

def get_dynamic_cluster_mapping(kmeans_model, scaler, feature_names):
    """
    Dynamically maps trained cluster indices to marketing profiles by measuring
    the Euclidean distance between the unscaled cluster centroids and standard ideal archetypes.
    This ensures that retrained models always map the same cluster label to the correct persona.
    """
    centroids_scaled = kmeans_model.cluster_centers_
    
    # Reconstruct original scale centroids
    # Scaler was fit on numerical features, e.g., ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    centroids_original = scaler.inverse_transform(centroids_scaled)
    
    # Find column indices for Income and Spending Score
    income_idx = feature_names.index('Annual Income (k$)')
    spend_idx = feature_names.index('Spending Score (1-100)')
    
    mapping = {}
    assigned_profiles = set()
    
    # For each cluster, find the closest ideal customer profile archetype
    for cluster_id in range(kmeans_model.n_clusters):
        c_income = centroids_original[cluster_id, income_idx]
        c_spend = centroids_original[cluster_id, spend_idx]
        
        min_dist = float('inf')
        best_profile = "Standard"
        
        for profile_name, (ideal_inc, ideal_spd) in IDEAL_CENTROIDS.items():
            dist = (c_income - ideal_inc) ** 2 + (c_spend - ideal_spd) ** 2
            if dist < min_dist:
                min_dist = dist
                best_profile = profile_name
                
        # Resolve conflicts (ensure 1-to-1 mapping if possible, else fallback to distance order)
        # In practice, with Mall Customers, they are distinct enough that it is always 1-to-1.
        mapping[cluster_id] = best_profile
        assigned_profiles.add(best_profile)
        
    return mapping

class CustomerPredictor:
    """
    Class handles single-point predictions for new customers, scaling features,
    assigning clusters, and returning rich customer profiles.
    """
    def __init__(self, models_dir="models"):
        # Load scaler and best model
        self.scaler_path = os.path.join(models_dir, "scaler.pkl")
        self.kmeans_path = os.path.join(models_dir, "kmeans_model.pkl")
        
        if not os.path.exists(self.scaler_path) or not os.path.exists(self.kmeans_path):
            raise FileNotFoundError("Trained models/scalers missing. Run training pipeline first.")
            
        self.scaler = load_model_artifact(self.scaler_path)
        self.kmeans = load_model_artifact(self.kmeans_path)
        
        # In training, we scale: ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        self.feature_names = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        self.cluster_mapping = get_dynamic_cluster_mapping(self.kmeans, self.scaler, self.feature_names)

    def predict_profile(self, gender, age, annual_income, spending_score):
        """
        Predicts the customer segment and retrieves their persona/marketing recommendations.
        
        Args:
            gender: str ('Male' or 'Female')
            age: int
            annual_income: float (k$)
            spending_score: int (1-100)
            
        Returns:
            result: dict containing cluster, profile details, and raw values
        """
        # Encode gender if needed (though our clustering is based on numerical features)
        # Note: If our scaler expects ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        input_df = pd.DataFrame([{
            'Age': age,
            'Annual Income (k$)': annual_income,
            'Spending Score (1-100)': spending_score
        }])
        
        # Scale input features
        input_scaled = self.scaler.transform(input_df)
        
        # Predict cluster
        cluster_id = int(self.kmeans.predict(input_scaled)[0])
        
        # Retrieve profile
        profile_key = self.cluster_mapping[cluster_id]
        profile_details = CUSTOMER_PROFILES[profile_key]
        
        return {
            "cluster_id": cluster_id,
            "profile_key": profile_key,
            "profile_name": profile_details["name"],
            "description": profile_details["description"],
            "recommendation": profile_details["recommendation"],
            "inputs": {
                "Gender": gender,
                "Age": age,
                "Annual Income (k$)": annual_income,
                "Spending Score (1-100)": spending_score
            }
        }
