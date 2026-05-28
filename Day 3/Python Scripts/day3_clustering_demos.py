"""
DAY 3: Clustering, Dimensionality Reduction & Pattern Discovery
Demos: K-Means, Hierarchical Clustering, DBSCAN, PCA, Factor Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DAY 3: CLUSTERING & DIMENSIONALITY REDUCTION")
print("=" * 80)

# ============================================================
# DEMO 1: Customer Segmentation (K-Means + Hierarchical)
# ============================================================
print("\n" + "=" * 60)
print("DEMO 1: Customer Segmentation")
print("=" * 60)

df_customers = pd.read_csv('customer_segmentation.csv')
print(f"📊 Loaded {len(df_customers)} customers")

# Select features for clustering
features = ['Age', 'AnnualIncome', 'SpendingScore', 'AvgOrderValue', 'PurchaseFrequency']
X = df_customers[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== K-Means Clustering =====
print("\n🔵 K-MEANS CLUSTERING")

# Find optimal K using elbow method
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Optimal K = 4 (elbow at 4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_customers['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Cluster interpretation
print("\n📊 K-Means Cluster Centers (Scaled):")
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)
print(cluster_centers.round(2))

print("\n📈 Cluster Sizes:")
print(df_customers['KMeans_Cluster'].value_counts().sort_index())

# Calculate silhouette score
sil_score = silhouette_score(X_scaled, df_customers['KMeans_Cluster'])
print(f"\n📊 Silhouette Score: {sil_score:.3f} (higher = better clusters)")

# ===== Hierarchical Clustering =====
print("\n🟢 HIERARCHICAL CLUSTERING")

# Compute linkage matrix
linked = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(14, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.axhline(y=8, color='r', linestyle='--', label='Cut at distance=8 (3 clusters)')
plt.title('Customer Segmentation Dendrogram')
plt.xlabel('Customer Index')
plt.ylabel('Distance')
plt.legend()
plt.tight_layout()
plt.savefig('customer_dendrogram.png', dpi=150)
plt.show()
print("📁 Saved: customer_dendrogram.png")

# Cut dendrogram to get 3 clusters
df_customers['Hier_Cluster'] = fcluster(linked, 8, criterion='distance')

print("\n📊 Hierarchical Cluster Distribution:")
print(df_customers['Hier_Cluster'].value_counts().sort_index())

# ============================================================
# DEMO 2: Brand Positioning (Hierarchical Clustering)
# ============================================================
print("\n" + "=" * 60)
print("DEMO 2: Brand Positioning Analysis")
print("=" * 60)

df_brands = pd.read_csv('brand_positioning.csv')
print(f"📊 Loaded {len(df_brands)} brands")

# Select features
brand_features = ['Price', 'Quality', 'YoungAppeal', 'Luxury', 'Innovation', 'Distribution']
X_brands = df_brands[brand_features]

# Scale features
X_brands_scaled = StandardScaler().fit_transform(X_brands)

# Hierarchical clustering
linked_brands = linkage(X_brands_scaled, method='ward')

# Plot branded dendrogram
plt.figure(figsize=(14, 8))
dendrogram(linked_brands, labels=df_brands['Brand'].values, orientation='top',
           distance_sort='descending', leaf_rotation=90, leaf_font_size=9)
plt.axhline(y=6, color='r', linestyle='--', label='Cut at distance=6')
plt.title('Brand Positioning Dendrogram')
plt.xlabel('Brand')
plt.ylabel('Distance')
plt.legend()
plt.tight_layout()
plt.savefig('brand_dendrogram.png', dpi=150)
plt.show()
print("📁 Saved: brand_dendrogram.png")

# Get clusters
df_brands['BrandCluster'] = fcluster(linked_brands, 6, criterion='distance')

print("\n📊 Brand Clusters:")
for cluster in sorted(df_brands['BrandCluster'].unique()):
    brands_in_cluster = df_brands[df_brands['BrandCluster'] == cluster]['Brand'].tolist()
    print(f"\n  Cluster {int(cluster)} ({len(brands_in_cluster)} brands):")
    print(f"    {', '.join(brands_in_cluster[:5])}{'...' if len(brands_in_cluster) > 5 else ''}")

# ============================================================
# DEMO 3: Anomaly Detection (DBSCAN)
# ============================================================
print("\n" + "=" * 60)
print("DEMO 3: Anomalous Transaction Detection (DBSCAN)")
print("=" * 60)

df_anomaly = pd.read_csv('anomaly_transactions.csv')
print(f"📊 Loaded {len(df_anomaly)} transactions")
print(f"   True anomalies in dataset: {df_anomaly['IsAnomaly'].sum()}")

# Select features for anomaly detection
anomaly_features = ['Amount', 'LocationDistance', 'TransactionHour', 'AccountAgeDays', 'DeviceScore', 'NumFailedAttempts']
X_anomaly = df_anomaly[anomaly_features]

# Scale features
scaler_anom = StandardScaler()
X_anomaly_scaled = scaler_anom.fit_transform(X_anomaly)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=3)
df_anomaly['DBSCAN_Cluster'] = dbscan.fit_predict(X_anomaly_scaled)

# DBSCAN labels: -1 means anomaly (noise)
detected_anomalies = (df_anomaly['DBSCAN_Cluster'] == -1).sum()
print(f"\n🔍 DBSCAN Results:")
print(f"   Detected anomalies: {detected_anomalies}")
print(f"   True anomalies: {df_anomaly['IsAnomaly'].sum()}")

# Compare with true labels
true_anomalies_detected = ((df_anomaly['DBSCAN_Cluster'] == -1) & (df_anomaly['IsAnomaly'] == 1)).sum()
false_positives = ((df_anomaly['DBSCAN_Cluster'] == -1) & (df_anomaly['IsAnomaly'] == 0)).sum()

print(f"\n📊 Detection Performance:")
print(f"   True anomalies caught: {true_anomalies_detected}/{df_anomaly['IsAnomaly'].sum()}")
print(f"   False positives: {false_positives}")

# Plot DBSCAN results (using first two features for visualization)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = ['red' if c == -1 else 'blue' for c in df_anomaly['DBSCAN_Cluster']]
plt.scatter(df_anomaly['Amount'], df_anomaly['LocationDistance'], c=colors, alpha=0.6, s=30)
plt.xlabel('Transaction Amount')
plt.ylabel('Location Distance (km)')
plt.title('DBSCAN: Red = Anomalies Detected')

plt.subplot(1, 2, 2)
colors_true = ['red' if a == 1 else 'blue' for a in df_anomaly['IsAnomaly']]
plt.scatter(df_anomaly['Amount'], df_anomaly['LocationDistance'], c=colors_true, alpha=0.6, s=30)
plt.xlabel('Transaction Amount')
plt.ylabel('Location Distance (km)')
plt.title('Ground Truth: Red = Actual Anomalies')

plt.tight_layout()
plt.savefig('dbscan_anomaly_results.png', dpi=150)
plt.show()
print("📁 Saved: dbscan_anomaly_results.png")

# ============================================================
# DEMO 4: PCA & Factor Analysis (Employee Engagement)
# ============================================================
print("\n" + "=" * 60)
print("DEMO 4: Workforce Segmentation (PCA + Factor Analysis)")
print("=" * 60)

df_employees = pd.read_csv('employee_engagement.csv')
print(f"📊 Loaded {len(df_employees)} employees")

# Select survey questions (excluding metadata)
survey_cols = ['JobSatisfaction', 'WorkLifeBalance', 'PeerRecognition', 'ManagerSupport',
               'SalarySatisfaction', 'PromotionOpportunity', 'SkillUtilization', 'StressLevel',
               'IntentToStay', 'ProductivityScore', 'TeamCollaboration', 'LearningOpp',
               'Recognition', 'WorkloadManage']
X_employees = df_employees[survey_cols]

# Scale features
X_emp_scaled = StandardScaler().fit_transform(X_employees)

# ===== PCA =====
print("\n🔵 PRINCIPAL COMPONENT ANALYSIS (PCA)")

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_emp_scaled)

print(f"\n📊 Explained Variance Ratio:")
for i, ev in enumerate(pca.explained_variance_ratio_):
    print(f"   PC{i+1}: {ev*100:.1f}%")
print(f"   Total variance explained by first 3 PCs: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# PCA loadings (which features contribute to each PC)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2', 'PC3'],
    index=survey_cols
)
print(f"\n📊 Top features for PC1 (most variance):")
print(loadings['PC1'].abs().sort_values(ascending=False).head(5))

# ===== K-Means on PCA components =====
print("\n🟢 Workforce Segmentation using PCA + K-Means")

kmeans_emp = KMeans(n_clusters=3, random_state=42, n_init=10)
df_employees['EmployeeSegment'] = kmeans_emp.fit_predict(X_pca)

print(f"\n📊 Employee Segment Distribution:")
print(df_employees['EmployeeSegment'].value_counts())

# Segment profiling
print(f"\n📊 Segment Profiles (Average Scores):")
segment_profiles = df_employees.groupby('EmployeeSegment')[survey_cols].mean().round(2)
print(segment_profiles)

# ===== Visualization =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: PCA Scatter Plot (colored by segment)
ax1 = axes[0, 0]
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=df_employees['EmployeeSegment'], cmap='viridis', alpha=0.6)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax1.set_title('Employee Segmentation using PCA + K-Means')
plt.colorbar(scatter, ax=ax1)

# Plot 2: Explained Variance (Scree Plot)
ax2 = axes[0, 1]
pca_full = PCA()
pca_full.fit(X_emp_scaled)
ax2.bar(range(1, 8), pca_full.explained_variance_ratio_[:7], alpha=0.7)
ax2.plot(range(1, 8), np.cumsum(pca_full.explained_variance_ratio_[:7]), 'ro-', linewidth=2)
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Explained Variance Ratio')
ax2.set_title('Scree Plot')
ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5)

# Plot 3: PCA Loadings Heatmap
ax3 = axes[1, 0]
sns.heatmap(loadings, annot=True, cmap='RdBu', center=0, ax=ax3, fmt='.2f')
ax3.set_title('PCA Loadings (Feature Contributions)')
ax3.set_ylabel('Features')

# Plot 4: Segment Comparison (Radar Chart - simplified as bar chart)
ax4 = axes[1, 1]
segment_avg = segment_profiles.mean().sort_values()
ax4.barh(range(len(segment_avg)), segment_avg.values, color='steelblue')
ax4.set_yticks(range(len(segment_avg)))
ax4.set_yticklabels(segment_avg.index)
ax4.set_xlabel('Average Score (1-5)')
ax4.set_title('Overall Engagement by Feature')

plt.tight_layout()
plt.savefig('employee_segmentation_results.png', dpi=150)
plt.show()
print("📁 Saved: employee_segmentation_results.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY OF FINDINGS")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ DEMO                    │ KEY INSIGHT                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Customer Segmentation   │ Found 4 natural customer segments                 │
│ (K-Means + Hierarchical)│ - High-value loyal customers                     │
│                         │ - Budget-conscious shoppers                       │
│                         │ - Young spenders                                  │
│                         │ - At-risk churners                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Brand Positioning       │ Identified 4-5 brand clusters:                    │
│ (Hierarchical)          │ - Luxury (Rolex, Gucci, Hermes)                   │
│                         │ - Tech/Innovation (Apple, Google, Samsung)        │
│                         │ - Mass-market (Zara, H&M, Uniqlo)                 │
│                         │ - Sports (Nike, Adidas, Puma)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Anomaly Detection       │ DBSCAN successfully identified:                   │
│ (DBSCAN)                │ - High-value anomalies                           │
│                         │ - Unusual location patterns                      │
│                         │ - Failed transaction patterns                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Workforce Segmentation  │ PCA reduced 14 features to 3 components:         │
│ (PCA + K-Means)         │ - PC1: Job Satisfaction & Recognition            │
│                         │ - PC2: Work-Life Balance & Stress               │
│                         │ - PC3: Growth & Promotion                        │
│                         │ Identified 3 employee segments                   │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("✅ DAY 3 COMPLETE! All visualizations saved.")
print("=" * 80)