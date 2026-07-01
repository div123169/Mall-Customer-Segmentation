import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Set page configurations
st.set_page_config(
    page_title="Mall Customer Segmentation Hub",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design and branding
st.markdown("""
<style>
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1e293b;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(120deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    .profile-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        color: #1e293b;
    }
    .vip-card {
        background-color: #f3e8ff;
        border-left: 6px solid #8b5cf6;
    }
    .saver-card {
        background-color: #e0f2fe;
        border-left: 6px solid #0284c7;
    }
    .impulsive-card {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
    }
    .frugal-card {
        background-color: #dcfce7;
        border-left: 6px solid #22c55e;
    }
    .standard-card {
        background-color: #f1f5f9;
        border-left: 6px solid #64748b;
    }
</style>
""", unsafe_allow_html=True)

# Imports from src
from src.utils import load_data, DATASET_PATH
from src.predict import CustomerPredictor, CUSTOMER_PROFILES

# Load helper functions with Streamlit caching for optimization
@st.cache_data
def get_cached_data():
    return load_data()

@st.cache_resource
def get_cached_predictor():
    return CustomerPredictor(models_dir="models")

# Initialize models and load dataset
try:
    df_raw = get_cached_data()
    predictor = get_cached_predictor()
except Exception as e:
    st.error(f"Error loading models or dataset. Please make sure the pipeline has been run first: {e}")
    st.stop()

# Helper function to get color map for profiles
PROFILE_COLORS = {
    "Affluent High-Spenders (VIP)": "#8b5cf6",
    "Affluent Savers (Careful)": "#0284c7",
    "Budget Impulsive Spenders (Careless)": "#ef4444",
    "Conservative Economizers (Sensible)": "#22c55e",
    "Balanced Middle Class (Standard)": "#64748b"
}

# Sidebar Content
st.sidebar.markdown("<h2 style='text-align: center; color: #1e3a8a;'>🛒 Mall Customer Hub</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation Menu", 
    [
        "📊 Customer Segment Finder", 
        "📈 Dataset Explorer", 
        "🏆 Model Performance & Info"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.info("""
- **Algorithm**: K-Means Clustering
- **No. of Clusters (K)**: 6
- **Scaling Method**: StandardScaler
- **Optimization Metric**: Silhouette Score
""")

# Expandable internship submission details in sidebar
with st.sidebar.expander("🎓 Internship Details"):
    st.write("**Project Title**:")
    st.write("Mall Customer Segmentation using Machine Learning")
    st.write("**Target submission**:")
    st.write("Internship Capstone Project")
    st.write("**Python Version**: 3.13.5")

# APP MODE: Customer Segment Finder
if app_mode == "📊 Customer Segment Finder":
    st.markdown("<h1 class='main-header'>Mall Customer Segmentation System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>A machine learning dashboard that groups mall customers based on demographics and buying habits, providing tailored marketing approaches.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Enter Customer Details")
        with st.form("customer_input_form"):
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Age (Years)", 18, 70, 35)
            annual_income = st.slider("Annual Income (k$)", 15, 140, 50)
            spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)
            
            submit_btn = st.form_submit_button("Find Customer Segment")
            
        # Trigger prediction (always runs once on load using default values, then updates on submit)
        prediction = predictor.predict_profile(gender, age, annual_income, spending_score)
        
    with col2:
        st.subheader("Customer Analytics & Profile")
        
        # Style card based on cluster type
        profile_key = prediction["profile_key"]
        card_class = f"profile-card {profile_key.lower()}-card"
        
        # Display Prediction Result Card
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="margin-top: 0; font-size: 1.4rem;">🎯 Segment: {prediction['profile_name']}</h3>
            <p><strong>Description:</strong> {prediction['description']}</p>
            <hr style="border-top: 1px solid rgba(0,0,0,0.1); margin: 1rem 0;">
            <h4 style="margin-bottom: 0.5rem; font-size: 1.1rem; font-weight: 600;">💡 Actionable Marketing Strategy:</h4>
            <p>{prediction['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Small Key-Value Metrics Summary for user input
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Input Annual Income", f"${annual_income}k")
        with m_col2:
            st.metric("Input Spending Score", f"{spending_score}/100")
        with m_col3:
            st.metric("Assigned Cluster ID", f"Cluster {prediction['cluster_id']}")

    st.markdown("---")
    st.subheader("Cluster Visualization (Real-Time Mapping)")
    
    # Run prediction for all rows in the dataset to show dynamic color matching
    @st.cache_data
    def get_dataset_with_profiles(_df, _predictor):
        df_copy = _df.copy()
        profiles = []
        for idx, row in df_copy.iterrows():
            res = _predictor.predict_profile(row['Gender'], row['Age'], row['Annual Income (k$)'], row['Spending Score (1-100)'])
            profiles.append(res['profile_name'])
        df_copy['Profile Name'] = profiles
        return df_copy
        
    df_with_p = get_dataset_with_profiles(df_raw, predictor)
    
    # 3D interactive Scatter Plot
    fig = px.scatter_3d(
        df_with_p, 
        x='Age', 
        y='Annual Income (k$)', 
        z='Spending Score (1-100)',
        color='Profile Name',
        color_discrete_map=PROFILE_COLORS,
        opacity=0.6,
        size_max=8,
        height=600
    )
    
    # Highlight the newly predicted/inputted customer in the 3D plot
    fig.add_trace(go.Scatter3d(
        x=[age],
        y=[annual_income],
        z=[spending_score],
        mode='markers',
        marker=dict(
            size=14,
            color='yellow',
            symbol='diamond',
            line=dict(width=3, color='black')
        ),
        name='Current Inquiry'
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Database Distribution & Statistics")
    
    stat_col1, stat_col2 = st.columns([1, 1.2])
    
    with stat_col1:
        # Pie chart of customer segments
        profile_counts = df_with_p['Profile Name'].value_counts().reset_index()
        profile_counts.columns = ['Segment', 'Count']
        
        pie_fig = px.pie(
            profile_counts, 
            values='Count', 
            names='Segment',
            color='Segment',
            color_discrete_map=PROFILE_COLORS,
            hole=0.4,
            title="Customer Distribution by Segment"
        )
        st.plotly_chart(pie_fig, use_container_width=True)
        
    with stat_col2:
        # Mean stats table
        st.write("#### Numerical Averages per Segment")
        summary_stats = df_with_p.groupby('Profile Name')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().reset_index()
        summary_stats.columns = ['Profile Name', 'Average Age', 'Average Income (k$)', 'Average Spend Score']
        
        # Format the table for neat styling
        summary_stats['Average Age'] = summary_stats['Average Age'].round(1)
        summary_stats['Average Income (k$)'] = summary_stats['Average Income (k$)'].round(1)
        summary_stats['Average Spend Score'] = summary_stats['Average Spend Score'].round(1)
        
        st.dataframe(summary_stats.set_index('Profile Name'), use_container_width=True)

# APP MODE: Dataset Explorer
elif app_mode == "📈 Dataset Explorer":
    st.title("Dataset Explorer")
    st.write("Detailed view of the Kaggle Mall Customers Dataset used to train our models.")
    
    st.subheader("Raw Data Table")
    st.dataframe(df_raw, use_container_width=True)
    
    # Detailed Statistics
    st.subheader("Key Demographic Distribution")
    
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        if os.path.exists("assets/income_vs_spend_scatter.png"):
            st.image("assets/income_vs_spend_scatter.png", caption="Income vs Spending Score Scatter Plot", use_container_width=True)
        else:
            st.warning("Scatter plot figure missing from assets folder. Train model first.")
            
    with eda_col2:
        if os.path.exists("assets/correlation_heatmap.png"):
            st.image("assets/correlation_heatmap.png", caption="Correlation Matrix Heatmap", use_container_width=True)
        else:
            st.warning("Correlation heatmap missing.")
            
    st.markdown("---")
    st.subheader("Individual Feature Histograms")
    
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        if os.path.exists("assets/age_distribution.png"):
            st.image("assets/age_distribution.png", caption="Age Distribution", use_container_width=True)
    with h_col2:
        if os.path.exists("assets/income_distribution.png"):
            st.image("assets/income_distribution.png", caption="Annual Income Distribution", use_container_width=True)
    with h_col3:
        if os.path.exists("assets/spending_score_distribution.png"):
            st.image("assets/spending_score_distribution.png", caption="Spending Score Distribution", use_container_width=True)

# APP MODE: Model Performance & Info
elif app_mode == "🏆 Model Performance & Info":
    st.title("Model Tuning & Clustering Performance")
    st.write("Compare the performance of the K-Means, Hierarchical (Agglomerative), and DBSCAN algorithms on the Mall Customers dataset.")
    
    comparison_path = "models/model_comparison.csv"
    if os.path.exists(comparison_path):
        st.subheader("Evaluation Metrics Table")
        comp_df = pd.read_csv(comparison_path, index_col=0)
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.warning("Model comparison data is missing. Please run the training pipeline to generate it.")
        
    st.markdown("---")
    st.subheader("Hyperparameter Optimization Curves (K-Means)")
    
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        if os.path.exists("assets/kmeans_elbow_curve.png"):
            st.image("assets/kmeans_elbow_curve.png", caption="Elbow Curve (Within-Cluster Sum of Squares)", use_container_width=True)
        else:
            st.warning("Elbow Curve missing.")
            
    with opt_col2:
        if os.path.exists("assets/kmeans_silhouette_scores.png"):
            st.image("assets/kmeans_silhouette_scores.png", caption="Silhouette Scores for varying K", use_container_width=True)
        else:
            st.warning("Silhouette scores missing.")
            
    st.markdown("---")
    st.subheader("Clustering Algorithm Outputs")
    
    out_col1, out_col2 = st.columns(2)
    with out_col1:
        if os.path.exists("assets/kmeans_clusters_2d.png"):
            st.image("assets/kmeans_clusters_2d.png", caption="K-Means Clusters (Income vs Spend Projection)", use_container_width=True)
    with out_col2:
        if os.path.exists("assets/kmeans_clusters_pca.png"):
            st.image("assets/kmeans_clusters_pca.png", caption="PCA Dimensionality Reduction Projection (2D)", use_container_width=True)
