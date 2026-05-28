"""
DAY 1 - DEMO 1: Customer Churn Prediction
Business Problem: Which customers are likely to cancel their subscription?
Algorithms: Logistic Regression, Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)

print("=" * 80)
print("DEMO 1: Customer Churn Prediction")
print("=" * 80)

# ========== STEP 1: LOAD DATA ==========
print("\n📂 Loading customer data...")
df = pd.read_excel('D:\\Academics\\MDPs\\5-Day Online Hands-on Workshop (MDP) on AIML\\Day 1\\Datasets\\churn_data.xlsx')
print(f"✅ Loaded {len(df)} customer records")
print(f"   Features: {df.shape[1] - 2}")  # Excluding CustomerID and Churn
print(f"   Churn rate: {df['Churn'].mean()*100:.1f}%")

print("\n📊 First 5 rows:")
print(df.head())

# ========== STEP 2: DATA PREPROCESSING ==========
print("\n" + "=" * 50)
print("STEP 2: Data Preprocessing")
print("=" * 50)

# Separate features and target
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\n📋 Categorical columns: {categorical_cols}")

# Encode categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    print(f"   Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Handle boolean columns (convert to int)
boolean_cols = ['HasOnlineSecurity', 'HasTechSupport']
for col in boolean_cols:
    if col in X.columns:
        X[col] = X[col].astype(int)

print(f"\n✅ Preprocessing complete. Final features: {X.shape[1]}")

# ========== STEP 3: EXPLORATORY DATA ANALYSIS ==========
print("\n" + "=" * 50)
print("STEP 3: Exploratory Data Analysis")
print("=" * 50)

# Churn distribution
print("\n📊 Target Distribution:")
print(df['Churn'].value_counts())
print(f"Churn Percentage: {df['Churn'].mean()*100:.1f}%")

# Compare churners vs non-churners
print("\n📊 Feature Comparison (Churners vs Non-Churners):")
numeric_features = ['Age', 'TenureMonths', 'MonthlyCharges', 'NumSupportTickets']
for feature in numeric_features:
    churn_mean = df[df['Churn']==1][feature].mean()
    no_churn_mean = df[df['Churn']==0][feature].mean()
    print(f"   {feature}: Churners={churn_mean:.1f} | Non-Churners={no_churn_mean:.1f}")

# ========== STEP 4: TRAIN-TEST SPLIT ==========
print("\n" + "=" * 50)
print("STEP 4: Train-Test Split")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training set: {len(X_train)} samples")
print(f"✅ Testing set: {len(X_test)} samples")
print(f"   Training churn rate: {y_train.mean()*100:.1f}%")
print(f"   Testing churn rate: {y_test.mean()*100:.1f}%")

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== STEP 5: TRAIN LOGISTIC REGRESSION ==========
print("\n" + "=" * 50)
print("STEP 5: Logistic Regression Model")
print("=" * 50)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predictions
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("\n📈 Logistic Regression Performance:")
print(f"   Accuracy:  {accuracy_score(y_test, lr_pred):.3f}")
print(f"   Precision: {precision_score(y_test, lr_pred):.3f}")
print(f"   Recall:    {recall_score(y_test, lr_pred):.3f}")
print(f"   F1 Score:  {f1_score(y_test, lr_pred):.3f}")
print(f"   AUC-ROC:   {roc_auc_score(y_test, lr_proba):.3f}")

# Feature importance (coefficients)
print("\n📊 Logistic Regression Coefficients (Top predictors of churn):")
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', ascending=False)
for idx, row in coefficients.head(5).iterrows():
    direction = "INCREASES" if row['Coefficient'] > 0 else "DECREASES"
    print(f"   {row['Feature']}: {row['Coefficient']:.3f} ({direction} churn probability)")

# ========== STEP 6: TRAIN RANDOM FOREST ==========
print("\n" + "=" * 50)
print("STEP 6: Random Forest Model")
print("=" * 50)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate
print("\n📈 Random Forest Performance:")
print(f"   Accuracy:  {accuracy_score(y_test, rf_pred):.3f}")
print(f"   Precision: {precision_score(y_test, rf_pred):.3f}")
print(f"   Recall:    {recall_score(y_test, rf_pred):.3f}")
print(f"   F1 Score:  {f1_score(y_test, rf_pred):.3f}")
print(f"   AUC-ROC:   {roc_auc_score(y_test, rf_proba):.3f}")

# Feature importance
print("\n📊 Random Forest Feature Importance (Top 5):")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.3f}")

# ========== STEP 7: MODEL COMPARISON ==========
print("\n" + "=" * 50)
print("STEP 7: Model Comparison")
print("=" * 50)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [accuracy_score(y_test, lr_pred), accuracy_score(y_test, rf_pred)],
    'Precision': [precision_score(y_test, lr_pred), precision_score(y_test, rf_pred)],
    'Recall': [recall_score(y_test, lr_pred), recall_score(y_test, rf_pred)],
    'F1 Score': [f1_score(y_test, lr_pred), f1_score(y_test, rf_pred)],
    'AUC-ROC': [roc_auc_score(y_test, lr_proba), roc_auc_score(y_test, rf_proba)]
})

print("\n📊 Performance Comparison:")
print(results.to_string(index=False))

# Determine best model
best_model = "Random Forest" if results.iloc[1]['AUC-ROC'] > results.iloc[0]['AUC-ROC'] else "Logistic Regression"
print(f"\n🏆 Best Model: {best_model}")

# ========== STEP 8: BUSINESS RECOMMENDATIONS ==========
print("\n" + "=" * 50)
print("STEP 8: Business Recommendations")
print("=" * 50)

# Calculate business impact
campaign_cost = 10  # $10 per retention offer
customer_lifetime_value = 500  # $500 average CLV

# Using Random Forest predictions
high_risk_customers = rf_proba >= 0.5
n_high_risk = high_risk_customers.sum()
actual_churn_in_high_risk = y_test[high_risk_customers].sum() if n_high_risk > 0 else 0

cost_of_campaign = n_high_risk * campaign_cost
revenue_saved = actual_churn_in_high_risk * customer_lifetime_value
net_benefit = revenue_saved - cost_of_campaign

print(f"\n💰 Business Impact Analysis:")
print(f"   Customers identified as high-risk: {n_high_risk}")
print(f"   Actual churners in high-risk group: {actual_churn_in_high_risk}")
print(f"   Campaign cost: ${cost_of_campaign:,.0f}")
print(f"   Revenue saved: ${revenue_saved:,.0f}")
print(f"   Net benefit: ${net_benefit:,.0f}")

print("\n📋 Actionable Insights:")
print("   1. Focus retention offers on customers with:")
top_features = feature_importance.head(3)['Feature'].tolist()
for f in top_features:
    print(f"      - High {f}")
print("   2. Implement early warning system for high-risk customers")
print("   3. Run A/B test on retention offers for validation")

# ========== STEP 9: VISUALIZATION ==========
print("\n" + "=" * 50)
print("STEP 9: Visualization")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Confusion Matrix - Random Forest
ax1 = axes[0, 0]
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix - Random Forest')

# Plot 2: ROC Curves
ax2 = axes[0, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
ax2.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_score(y_test, lr_proba):.3f})')
ax2.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_score(y_test, rf_proba):.3f})')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves Comparison')
ax2.legend()

# Plot 3: Feature Importance
ax3 = axes[1, 0]
top_features_plot = feature_importance.head(8)
ax3.barh(range(len(top_features_plot)), top_features_plot['Importance'].values, color='teal')
ax3.set_yticks(range(len(top_features_plot)))
ax3.set_yticklabels(top_features_plot['Feature'].values)
ax3.set_xlabel('Importance')
ax3.set_title('Top 8 Features for Churn Prediction')
ax3.invert_yaxis()

# Plot 4: Churn by Key Feature
ax4 = axes[1, 1]
key_feature = top_features[0]
churn_by_feature = df.groupby(pd.cut(df[key_feature], bins=5))['Churn'].mean()
ax4.bar(range(len(churn_by_feature)), churn_by_feature.values, color='coral')
ax4.set_xticks(range(len(churn_by_feature)))
ax4.set_xticklabels([f'{int(b.left)}-{int(b.right)}' for b in churn_by_feature.index], rotation=45)
ax4.set_xlabel(key_feature)
ax4.set_ylabel('Churn Rate')
ax4.set_title(f'Churn Rate by {key_feature}')

plt.tight_layout()
plt.savefig('churn_prediction_results.png', dpi=150)
plt.show()

print("\n✅ Visualizations saved as 'churn_prediction_results.png'")
print("\n" + "=" * 80)
print("✅ Customer Churn Prediction Complete!")
print("=" * 80)
