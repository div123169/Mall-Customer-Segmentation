# Mall Customer Segmentation using Machine Learning

## Intern Details

- **Intern ID:** CMPTBD7AX0
- **Full Name:** Divyansh Rai
- **No. of Weeks:** 2 Weeks
- **Project Name:** Mall Customer Segmentation
- **Project Scope:** This project segments mall customers into meaningful groups using machine learning clustering algorithms. It includes data preprocessing, exploratory data analysis, clustering model training, evaluation, and a Streamlit web application for real-time customer segmentation and marketing recommendations.



This repository contains a complete, production-ready, and intermediate-level Machine Learning project titled **"Mall Customer Segmentation using Machine Learning"**. 

This system applies unsupervised machine learning algorithms (K-Means, Hierarchical Agglomerative Clustering, and DBSCAN) to the real Mall Customers dataset from Kaggle to segment clients based on their demographic variables and purchasing behavior. It is packaged with modular code, exploratory analysis plots, pre-trained model artifacts, and an interactive Streamlit dashboard for real-time customer profiling and marketing recommendations.

---

## 📂 Project Structure

```
Mall/
│
├── assets/                                 # Visualization assets used by the Streamlit dashboard
│   ├── age_distribution.png
│   ├── boxplots_by_gender.png
│   ├── correlation_heatmap.png
│   ├── gender_distribution.png
│   ├── income_distribution.png
│   ├── income_vs_spend_scatter.png
│   ├── kmeans_clusters_2d.png
│   ├── kmeans_clusters_pca.png
│   ├── kmeans_elbow_curve.png
│   ├── kmeans_silhouette_scores.png
│   ├── pairplot.png
│   └── spending_score_distribution.png
│
├── dataset/                                # Mall Customers dataset
│   └── Mall_Customers.csv
│
├── models/                                 # Saved trained models and metadata
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── model_comparison.csv
│
├── notebooks/                              
│   ├── EDA.ipynb                           # Exploratory Data Analysis notebook
│   └── plots/                              # Plots generated during EDA and model training
│       ├── age_distribution.png
│       ├── boxplots_by_gender.png
│       ├── correlation_heatmap.png
│       ├── gender_distribution.png
│       ├── income_distribution.png
│       ├── income_vs_spend_scatter.png
│       ├── kmeans_clusters_2d.png
│       ├── kmeans_clusters_pca.png
│       ├── kmeans_elbow_curve.png
│       ├── kmeans_silhouette_scores.png
│       ├── pairplot.png
│       └── spending_score_distribution.png
│
├── screenshots/                            # Screenshots for README documentation
│   ├── dashboard.png
│   ├── cluster_visualization.png
│   ├── clustering_results.png
│   ├── dataset_table.png
│   ├── demographic_analysis.png
│   ├── evaluation_metrics.png
│   ├── feature_histograms.png
│   ├── input_form.png
│   ├── model_performance.png
│   └── prediction_result.png
│
├── src/                                    # Source code modules
│   ├── clustering.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── run_pipeline.py
├── README.md
└── report.md
```

---

## 🛠️ Technology Stack & Libraries

- **Programming Language**: Python 3.13.5
- **Data Engineering**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `joblib`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Web Interface**: `streamlit`
- **Development**: `nbformat`, `ipykernel`

---

## ⚙️ Installation & Setup Guide

Follow these steps to run the project locally on your machine.

### 1. Clone or Copy the Workspace
Navigate to the root directory `Mall/`.

### 2. Install Dependencies
Install all required libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Run the Development Pipeline
Run the central execution script `run_pipeline.py`. This script automatically:
1. Downloads the real dataset from Kaggle/GitHub RAW URL.
2. Cleans features, maps Genre to Gender, and standardizes variables.
3. Performs Exploratory Data Analysis (EDA) and saves 12 visualization plots to `notebooks/plots/`.
4. Performs a grid search to optimize $K$ via average Silhouette Score.
5. Trains and evaluates K-Means, Agglomerative Hierarchical, and DBSCAN.
6. Saves the best K-Means model, standard scaler, and comparative csv results.

```bash
python run_pipeline.py
```

### 4. Start the Interactive Streamlit Web Application
Launch the client-facing dashboard to profile customers and generate personalized marketing campaigns in real-time:
```bash
streamlit run app.py
```
After executing this command, Streamlit will open the application in your default web browser (typically at `http://localhost:8501`).

---

## 📊 Key Customer Persona Categories

The customer segments are dynamically mapped based on their centroids to guarantee consistent behavioral personas across training sessions:
1. **Affluent High-Spenders (VIP)**: High Income, High Spending. Premium brand targets.
2. **Affluent Savers (Careful)**: High Income, Low Spending. Quality and utility targets.
3. **Budget Impulsive Spenders (Careless)**: Low Income, High Spending. Trend and flash sale targets.
4. **Conservative Economizers (Sensible)**: Low Income, Low Spending. Value-packs and bargain targets.
5. **Balanced Middle Class (Standard)**: Medium Income, Medium Spending. Steady repeat shoppers.

---

## 🎓 Internship Submission Details
- **Project Title**: Mall Customer Segmentation using Machine Learning
- **Developer**: Machine Learning Engineering Intern
- **Status**: Completed and ready for submission
- **Format**: Clean, modular, document-backed codebase adhering to PEP-8 principles.
