# Mall Customer Segmentation Project Report

**Title**: Customer Behavioral Segmentation using Unsupervised Machine Learning  
**Author**: Machine Learning Engineering Intern  
**Objective**: Build a robust, client-facing customer segmentation model for Mall Business Analytics  
**Date**: June 2026  

---

## 1. Executive Summary
Customer segmentation is a critical process for modern retail environments, enabling businesses to divide their customer base into distinct cohorts based on purchasing behaviors and demographics. This report details the development of an intermediate-level customer segmentation pipeline utilizing the **Mall Customers Dataset** (200 records). 

Through unsupervised learning, we compared **K-Means**, **Agglomerative Hierarchical Clustering**, and **DBSCAN**. Optimization of K-Means using the Elbow Method and Silhouette Analysis identified **$K=6$** as the optimal number of clusters for our multi-dimensional feature space (`Age`, `Annual Income (k$)`, and `Spending Score (1-100)`). A client-facing Streamlit dashboard was successfully implemented to categorize inquiries in real-time, providing tailored marketing campaigns.

---

## 2. Introduction & Problem Statement
Understanding the demographics and purchasing habits of store clients is key to improving customer retention, maximizing average cart value, and launching cost-efficient marketing campaigns.

### Dataset Schema
The dataset consists of $N=200$ customer profiles with 5 attributes:
1. `CustomerID`: Unique numerical identification (dropped during modeling).
2. `Gender`: Categorical variable (Female / Male).
3. `Age`: Numerical variable indicating client's age (18 to 70).
4. `Annual Income (k$)`: Numerical variable representing annual earnings (15k to 137k).
5. `Spending Score (1-100)`: Numerical score (1 to 99) assigned by the mall based on purchase frequency and behavior.

---

## 3. Exploratory Data Analysis (EDA) Insights
Detailed analysis of the variables revealed the following key properties of the customer base:
- **Demographics**: Females represent 56% of the customer base, outnumbering males (44%).
- **Age Distribution**: The customer base is right-skewed, peaking heavily between ages 30 and 35. 
- **Income Distribution**: Most customers earn between $40k and $80k annually, with a maximum salary of $137k.
- **Spending Score Distribution**: The score peaks around 50, showing a normal-like distribution with secondary peaks at extremes.
- **Correlations**:
  - A moderate negative correlation ($-0.33$) exists between Age and Spending Score, suggesting that younger shoppers are generally more active spenders.
  - Annual Income shows negligible linear correlation with Age ($-0.01$) and Spending Score ($0.01$). This confirms that spending habits are independent of income alone, indicating that *unsupervised clustering* is necessary to segment customers.

---

## 4. Preprocessing & Feature Engineering
Before running clustering algorithms, we applied a strict preprocessing pipeline:
1. **Cleaning**: Duplicate records check returned $0$ duplicates. Missing value analysis returned $0$ null values.
2. **Gender Encoding**: The categorical variable `Gender` was mapped to binary numerical space (Female -> 0, Male -> 1) to enable inclusion in correlation matrices.
3. **Standardization**: Because distance-based clustering algorithms (like K-Means and Hierarchical Clustering) are sensitive to scaling, numeric variables (`Age`, `Annual Income`, `Spending Score`) were scaled using `StandardScaler` to have a mean of 0 and standard deviation of 1.

$$\tilde{x} = \frac{x - \mu}{\sigma}$$

---

## 5. Model Training & Evaluation
We compared K-Means, Agglomerative Hierarchical, and DBSCAN algorithms.

### 5.1 K-Means Hyperparameter Tuning
To optimize the number of clusters $K$ for K-Means, we calculated Within-Cluster Sum of Squares (WCSS) and Silhouette Scores for $K \in [2, 10]$:
- **Elbow Method**: Plotted WCSS against K. The curve showed a visible bend (elbow) around $K=5$ and $K=6$.
- **Silhouette Score**: Evaluated average silhouette coefficients (measuring cluster cohesion vs. separation). The silhouette score peaked at **$K=6$** with a score of **$0.428$** in the 3D feature space.

### 5.2 Model Comparison Table
The algorithms were trained on the preprocessed 3D feature array. The results are summarized below:

| Algorithm | Number of Clusters | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
| :--- | :---: | :---: | :---: | :---: |
| **K-Means** | **6** | **0.4286** | **0.9015** | **135.10** |
| **Hierarchical (Agglomerative)** | 6 | 0.3814 | 0.9856 | 127.99 |
| **DBSCAN (eps=0.5, min_samples=5)** | 6 (excluding noise) | 0.1174 | 2.1154 | 117.80 |

### 5.3 Selection Rationale
- **K-Means** significantly outperformed Agglomerative Clustering and DBSCAN on both the **Silhouette Score** (closer to 1 is better) and the **Davies-Bouldin Index** (closer to 0 is better).
- DBSCAN performed poorly due to the relatively low density and uniform dispersion of some data points, labeling several points as noise ($-1$).
- Thus, K-Means was selected as our final production model.

---

## 6. Business Persona Mapping & Marketing Strategies
By running our distance-based centroid classifier on the K-Means clusters, we identified 5 distinct marketing profiles. (Since $K=6$, two sub-clusters representing older and younger shoppers mapped to the "Frugal" and "Standard" categories, capturing granular age demographics).

### 1. Affluent High-Spenders (VIP)
- **Centroid**: Income $\approx \$86k$, Spending Score $\approx 82$.
- **Characteristics**: High-earning, high-spending shoppers. Mostly middle-aged.
- **Marketing Strategy**: Launch premium rewards programs, provide early access to new lines, assign dedicated personal shoppers, and send invitations to luxury product launches.

### 2. Affluent Savers (Careful)
- **Centroid**: Income $\approx \$88k$, Spending Score $\approx 17$.
- **Characteristics**: High-earning but highly conservative spenders.
- **Marketing Strategy**: Focus on product quality, durability, longevity, and investment value. Target with utility-based advertisements and transparent cashback rewards rather than coupon deals.

### 3. Budget Impulsive Spenders (Careless)
- **Centroid**: Income $\approx \$25k$, Spending Score $\approx 78$.
- **Characteristics**: Low-income but high-spending individuals. Skewed heavily towards younger demographics.
- **Marketing Strategy**: Leverage social media influencer campaigns, promote limited-time discount codes, run flash sales, and integrate Buy-Now-Pay-Later (BNPL) payment plans.

### 4. Conservative Economizers (Sensible)
- **Centroid**: Income $\approx \$26k$, Spending Score $\approx 20$.
- **Characteristics**: Low-income and conservative spenders.
- **Marketing Strategy**: Promote essential utility bundles, budget brand versions, BOGO (Buy One Get One Free) deals, and straight price-reduction clearances.

### 5. Balanced Middle Class (Standard)
- **Centroid**: Income $\approx \$55k$, Spending Score $\approx 49$.
- **Characteristics**: Moderate income and spending habits. Represents the stable baseline customer.
- **Marketing Strategy**: Enroll in standard loyalty subscription programs, send personalized birthday coupons, and distribute seasonal newsletters to drive repeat store visits.

---

## 7. App Deployment Details
The system was deployed as an interactive Streamlit application (`app.py`), which includes:
- **Interactive Forms**: Allowing marketers to input a single client's Gender, Age, Income, and Spending Score.
- **Real-Time Predictions**: Categorizes the client and renders custom, color-coded HTML cards outlining the persona and marketing recommendations.
- **Interactive 3D Visualizations**: Uses Plotly to render the 3D cluster field and overlays the client's current inquiry as a glowing yellow diamond, visually situating them among the database cohorts.
- **Dynamic Metrics**: Summarizes database distributions via pie charts and mean tables.

---

## 8. Conclusion
The "Mall Customer Segmentation" system provides a complete end-to-end analytics pipeline that bridges machine learning with actionable marketing campaigns. By shifting from gut-feeling promotions to data-driven customer personas, the mall management can optimize advertising spend, improve customer satisfaction, and drive higher store revenues.

### Future Recommendations
1. **Dynamic Pricing Integration**: Integrate the segment finder with dynamic discount allocation systems.
2. **Online Behavior Tracking**: Connect the model with digital e-commerce tracking (e.g. cart abandonment rates) to enrich features beyond simple mall demographic files.
3. **Transition to Semi-Supervised Learning**: As customer feedback is gathered on the marketing campaigns, labeled data can be used to train active supervised classifiers (e.g., Random Forests) to predict segment response rates.
