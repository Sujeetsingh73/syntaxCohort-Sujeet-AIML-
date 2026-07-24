import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================================================
# STEP 1: Generate Synthetic Customer Dataset
# =========================================================
np.random.seed(42)
n_samples = 300

# Creating synthetic customer behavior data
annual_income = np.random.normal(60, 25, n_samples)          # In $1,000s
spending_score = np.random.normal(50, 20, n_samples)         # 1 to 100 scale
purchase_frequency = np.random.poisson(lam=12, size=n_samples) # Purchases per year
recency_days = np.random.randint(1, 365, n_samples)          # Days since last order

df_customer = pd.DataFrame({
    'AnnualIncome_k': np.clip(annual_income, 15, 150),
    'SpendingScore': np.clip(spending_score, 1, 100),
    'PurchaseFrequency': purchase_frequency,
    'RecencyDays': recency_days
})

print("--- RAW CUSTOMER DATASET ---")
print(df_customer.head())
print("\n" + "="*50 + "\n")

# =========================================================
# STEP 2: Standardize Data & PCA (Dimensionality Reduction)
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_customer)

# Reduce 4 features into 2 Principal Components for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("--- PCA EXPLAINED VARIANCE RATIO ---")
print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Total Variance Preserved: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
print("\n" + "="*50 + "\n")

# =========================================================
# STEP 3: Find Optimal Clusters using Elbow Method
# =========================================================
wcss = []  # Within-Cluster Sum of Squares
for i in range(1, 11):
    kmeans_test = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans_test.fit(X_scaled)
    wcss.append(kmeans_test.inertia_)

# Plotting Elbow Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='purple')
plt.title("Elbow Method For Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True)

# =========================================================
# STEP 4: K-Means Clustering Implementation
# =========================================================
# Based on elbow, let's select K = 3 clusters
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataset
df_customer['Cluster'] = cluster_labels

# Calculate Silhouette Score (Measures cluster quality: closer to 1 is better)
score = silhouette_score(X_scaled, cluster_labels)
print(f"--- K-MEANS EVALUATION ---")
print(f"Silhouette Score for K={optimal_k}: {score:.4f}")

# =========================================================
# STEP 5: Visualizing Clusters using PCA Projection
# =========================================================
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, edgecolors='k')
plt.title("Customer Segments (2D PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# STEP 6: Customer Profile Summary
# =========================================================
print("\n" + "="*50 + "\n")
print("--- CUSTOMER SEGMENT PROFILE SUMMARY (MEAN VALUES) ---")
segment_summary = df_customer.groupby('Cluster').mean()
print(segment_summary)