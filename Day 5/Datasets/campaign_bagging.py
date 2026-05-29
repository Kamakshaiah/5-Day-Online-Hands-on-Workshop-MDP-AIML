# ============================================================
# BAGGING ENSEMBLE - CAMPAIGN RESPONSE PREDICTION
# Dataset: campaign_data.xlsx
# Target: Responded (0 = No Response, 1 = Will Respond)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BAGGING ENSEMBLE - CAMPAIGN RESPONSE PREDICTION")
print("=" * 80)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n📂 STEP 1: Loading Data")
print("-" * 50)

df = pd.read_excel('campaign_data.xlsx')
print(f"✅ Data loaded successfully")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check target distribution
print(f"\nTarget Distribution (Responded):")
print(df['Responded'].value_counts())
print(f"Response rate: {df['Responded'].mean()*100:.2f}%")

# Check for missing values
print(f"\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 50)
print("STEP 2: Data Preprocessing")
print("=" * 50)

# Drop CustomerID (not a feature)
X = df.drop(['CustomerID', 'Responded'], axis=1)
y = df['Responded']

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    print(f"  Encoded {col}")

# Handle boolean columns (convert to int)
boolean_cols = ['IsWeekendPurchase', 'HasActiveSubscription']
for col in boolean_cols:
    if col in X.columns:
        X[col] = X[col].astype(int)

# Store feature names for later use
feature_names = X.columns.tolist()
print(f"\nTotal features: {len(feature_names)}")
print(f"Feature list: {feature_names}")

# Handle missing values if any
if X.isnull().sum().sum() > 0:
    print("\n⚠️ Missing values found. Filling with median...")
    X = X.fillna(X.median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Train response rate: {y_train.mean()*100:.1f}%")
print(f"Test response rate: {y_test.mean()*100:.1f}%")

# ============================================================
# STEP 3: TRAIN BASE MODEL (Decision Tree for comparison)
# ============================================================
print("\n" + "=" * 50)
print("STEP 3: Baseline Model (Single Decision Tree)")
print("=" * 50)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
y_proba_dt = dt_model.predict_proba(X_test)[:, 1]

print("\n📊 SINGLE DECISION TREE PERFORMANCE:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_dt):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_dt):.4f}")
print(f"  AUC-ROC:   {roc_auc_score(y_test, y_proba_dt):.4f}")

# ============================================================
# STEP 4: TRAIN BAGGING ENSEMBLE (Decision Tree as base estimator)
# ============================================================
print("\n" + "=" * 50)
print("STEP 4: Bagging Ensemble (100 Decision Trees)")
print("=" * 50)

# Initialize Bagging Classifier
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,        # Number of trees
    max_samples=0.8,         # Use 80% of samples for each tree
    max_features=0.8,        # Use 80% of features for each tree
    bootstrap=True,          # Sample with replacement
    bootstrap_features=False,# Don't bootstrap features
    n_jobs=-1,               # Use all CPU cores
    random_state=42
)

print("Bagging Parameters:")
print(f"  - Base Estimator: Decision Tree")
print(f"  - Number of Estimators: {bagging_model.n_estimators}")
print(f"  - Max Samples: {bagging_model.max_samples}")
print(f"  - Max Features: {bagging_model.max_features}")
print(f"  - Bootstrap: {bagging_model.bootstrap}")

# Train the model
print("\nTraining Bagging Ensemble...")
bagging_model.fit(X_train, y_train)
print("✅ Training complete")

# ============================================================
# STEP 5: MAKE PREDICTIONS
# ============================================================
print("\n" + "=" * 50)
print("STEP 5: Making Predictions")
print("=" * 50)

y_train_pred = bagging_model.predict(X_train)
y_test_pred = bagging_model.predict(X_test)
y_test_proba = bagging_model.predict_proba(X_test)[:, 1]

print(f"Training predictions: {len(y_train_pred)}")
print(f"Test predictions: {len(y_test_pred)}")

# ============================================================
# STEP 6: EVALUATION METRICS
# ============================================================
print("\n" + "=" * 50)
print("STEP 6: Model Evaluation")
print("=" * 50)

# Training metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, bagging_model.predict_proba(X_train)[:, 1])

# Test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print("\n📊 TRAINING SET PERFORMANCE:")
print(f"  - Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  - Precision: {train_precision:.4f}")
print(f"  - Recall:    {train_recall:.4f}")
print(f"  - F1 Score:  {train_f1:.4f}")
print(f"  - AUC-ROC:   {train_auc:.4f}")

print("\n📊 TEST SET PERFORMANCE:")
print(f"  - Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  - Precision: {test_precision:.4f}")
print(f"  - Recall:    {test_recall:.4f}")
print(f"  - F1 Score:  {test_f1:.4f}")
print(f"  - AUC-ROC:   {test_auc:.4f}")

# ============================================================
# STEP 7: CONFUSION MATRIX
# ============================================================
print("\n" + "=" * 50)
print("STEP 7: Confusion Matrix")
print("=" * 50)

cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print("                 Predicted")
print("              No Response  Response")
print("-" * 45)
print(f"Actual No Response  |     {cm[0,0]}          {cm[0,1]}")
print(f"Actual Response     |     {cm[1,0]}          {cm[1,1]}")

print(f"\n📊 Detailed Breakdown:")
print(f"  - True Negatives (Correctly predicted No Response): {cm[0,0]}")
print(f"  - False Positives (Wrongly predicted Response): {cm[0,1]}")
print(f"  - False Negatives (Missed Response): {cm[1,0]}")
print(f"  - True Positives (Correctly predicted Response): {cm[1,1]}")

# ============================================================
# STEP 8: CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 50)
print("STEP 8: Classification Report")
print("=" * 50)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Response', 'Response']))

# ============================================================
# STEP 9: FEATURE IMPORTANCE (from Random Forest for interpretation)
# ============================================================
print("\n" + "=" * 50)
print("STEP 9: Feature Importance")
print("=" * 50)

# Since Bagging doesn't provide native feature importance, train a Random Forest for interpretation
rf_for_importance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_for_importance.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_for_importance.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Top 10 Most Important Features for Campaign Response:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")

# ============================================================
# STEP 10: CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 50)
print("STEP 10: Cross-Validation")
print("=" * 50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(bagging_model, X_scaled, y, cv=cv, scoring='roc_auc')

print(f"5-Fold Cross-Validation AUC Scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================================
# STEP 11: COMPARE WITH RANDOM FOREST
# ============================================================
print("\n" + "=" * 50)
print("STEP 11: Comparison with Random Forest")
print("=" * 50)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n📊 RANDOM FOREST PERFORMANCE:")
print(f"  - Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  - Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"  - Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"  - F1 Score:  {f1_score(y_test, y_pred_rf):.4f}")
print(f"  - AUC-ROC:   {roc_auc_score(y_test, y_proba_rf):.4f}")

print("\n📊 MODEL COMPARISON:")
print("-" * 60)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'AUC':<10}")
print("-" * 60)
print(f"{'Single Decision Tree':<20} {accuracy_score(y_test, y_pred_dt):<12.4f} {precision_score(y_test, y_pred_dt):<12.4f} {recall_score(y_test, y_pred_dt):<12.4f} {roc_auc_score(y_test, y_proba_dt):<10.4f}")
print(f"{'Bagging Ensemble':<20} {test_accuracy:<12.4f} {test_precision:<12.4f} {test_recall:<12.4f} {test_auc:<10.4f}")
print(f"{'Random Forest':<20} {accuracy_score(y_test, y_pred_rf):<12.4f} {precision_score(y_test, y_pred_rf):<12.4f} {recall_score(y_test, y_pred_rf):<12.4f} {roc_auc_score(y_test, y_proba_rf):<10.4f}")

# ============================================================
# STEP 12: SAMPLE PREDICTIONS
# ============================================================
print("\n" + "=" * 50)
print("STEP 12: Sample Predictions (First 10 Test Samples)")
print("=" * 50)

print("\n📊 Sample | Actual | Predicted | Probability | Correct?")
print("-" * 70)

for i in range(min(10, len(y_test))):
    actual = "Response" if y_test.iloc[i] == 1 else "No Response"
    pred = "Response" if y_test_pred[i] == 1 else "No Response"
    prob = y_test_proba[i] if y_test_pred[i] == 1 else 1 - y_test_proba[i]
    correct = "✅ YES" if y_test.iloc[i] == y_test_pred[i] else "❌ NO"
    print(f"  {i+1:6d} | {actual:11} | {pred:11} | {prob*100:5.1f}%      | {correct}")

# ============================================================
# STEP 13: BUSINESS COST ANALYSIS
# ============================================================
print("\n" + "=" * 50)
print("STEP 13: Business Cost Analysis")
print("=" * 50)

# Define costs
campaign_cost_per_customer = 5.0    # $5 to send campaign
revenue_per_response = 85.0          # $85 average revenue per response

# Calculate using Random Forest (best performing)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()

total_customers = len(y_test)
campaign_cost = (tp + fp) * campaign_cost_per_customer
revenue_generated = tp * revenue_per_response
net_profit = revenue_generated - campaign_cost

print(f"\n💰 FINANCIAL ANALYSIS (using Random Forest):")
print(f"  - Total test customers: {total_customers}")
print(f"  - Customers targeted: {tp + fp}")
print(f"  - Campaign cost: ${campaign_cost:,.0f}")
print(f"  - Revenue generated: ${revenue_generated:,.0f}")
print(f"  - Net profit: ${net_profit:,.0f}")

# Compare with targeting everyone
target_all_cost = total_customers * campaign_cost_per_customer
target_all_revenue = y_test.sum() * revenue_per_response
target_all_profit = target_all_revenue - target_all_cost

print(f"\n📊 COMPARISON WITH MASS MARKETING:")
print(f"  - Mass marketing profit: ${target_all_profit:,.0f}")
print(f"  - Model-based profit: ${net_profit:,.0f}")
print(f"  - Improvement: ${net_profit - target_all_profit:,.0f} ({((net_profit - target_all_profit)/abs(target_all_profit))*100:.1f}%)")

# ============================================================
# STEP 14: BUSINESS SUMMARY
# ============================================================
print("\n" + "=" * 50)
print("BUSINESS SUMMARY")
print("=" * 50)

# Determine model quality rating
if test_auc >= 0.9:
    quality = "EXCELLENT"
elif test_auc >= 0.8:
    quality = "GOOD"
elif test_auc >= 0.7:
    quality = "MODERATE"
else:
    quality = "POOR"

# Determine the best model
best_model = "Random Forest" if test_auc > roc_auc_score(y_test, y_proba_dt) else "Bagging"

print(f"""
📊 MODEL PERFORMANCE SUMMARY:

  • Target Variable: Responded (0 = No Response, 1 = Response)
  • Total Samples: {len(df)}
  • Training Samples: {len(X_train)}
  • Testing Samples: {len(X_test)}
  • Features Used: {len(feature_names)}
  • Response Rate in Test Set: {y_test.mean()*100:.1f}%

  • Best Model: {best_model}
  • Test AUC-ROC: {max(test_auc, roc_auc_score(y_test, y_proba_rf)):.4f}
  • Model Quality Rating: {quality}

🎯 KEY INSIGHTS:

  • Most important feature: {feature_importance.iloc[0]['Feature']}
  • Top 3 features: {feature_importance.iloc[0]['Feature']}, {feature_importance.iloc[1]['Feature']}, {feature_importance.iloc[2]['Feature']}
  • Bagging improved over single Decision Tree by {(test_auc - roc_auc_score(y_test, y_proba_dt))*100:.1f}% in AUC

📋 RECOMMENDATIONS:

  1. DEPLOY the {best_model} model for campaign targeting
  2. TARGET only customers with predicted response probability > 0.5
  3. SAVE {campaign_cost_per_customer * (fp + tn):,.0f} by not sending to non-responders
  4. RETRAIN model monthly with new campaign data
  5. MONITOR response rates and update thresholds

💰 EXPECTED ROI:
  • Reduction in campaign cost: {((target_all_cost - campaign_cost)/target_all_cost)*100:.0f}%
  • Increase in profit per campaign: ${net_profit - target_all_profit:,.0f}
""")

print("=" * 80)
print("✅ BAGGING ENSEMBLE COMPLETE!")
print("=" * 80)