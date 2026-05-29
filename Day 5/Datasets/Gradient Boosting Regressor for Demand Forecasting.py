# ============================================================
# GRADIENT BOOSTING CLASSIFIER - DEMAND CATEGORY PREDICTION
# Reads data from: demand_classifier.xlsx
# Target: Demand_Category (High/Medium/Low)
# FIXED: classification_report labels parameter
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GRADIENT BOOSTING CLASSIFIER - DEMAND CATEGORY PREDICTION")
print("=" * 80)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n📂 STEP 1: Loading Data")
print("-" * 50)

df = pd.read_excel('demand_data_corrected.xlsx')
print(f"✅ Data loaded successfully")
print(f"Dataset shape: {df.shape}")

# ============================================================
# STEP 2: OPTIONAL - COMBINE LOW WITH MEDIUM (RECOMMENDED)
# ============================================================
# Since Low has only 1 sample, combine it with Medium
print("\n📊 Combining 'Low' category with 'Medium' due to insufficient samples...")
df['Demand_Category'] = df['Demand_Category'].replace({'Low': 'Medium'})
print("  ✅ 'Low' merged into 'Medium'")

# ============================================================
# STEP 3: PREPARE FEATURES AND TARGET
# ============================================================
print("\n" + "=" * 50)
print("STEP 1: Preparing Features and Target")
print("=" * 50)

# Define target column
target = 'Demand_Category'
print(f"Target variable: {target}")

# Check target distribution
print(f"\nTarget Distribution:")
print(df[target].value_counts())

# Define features to EXCLUDE
exclude_cols = ['Date', 'Product_A_Demand', target]

# Select features
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"\nFeature columns ({len(feature_cols)} features):")
print(feature_cols)

X = df[feature_cols]
y = df[target]

# ============================================================
# STEP 4: ENCODE TARGET VARIABLE
# ============================================================
print("\n" + "=" * 50)
print("STEP 2: Encoding Target Variable")
print("=" * 50)

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"Target encoding:")
for i, class_name in enumerate(le_target.classes_):
    print(f"  {class_name} → {i}")

# Store the actual classes present in the data
actual_classes = le_target.classes_
print(f"\nClasses in data: {list(actual_classes)}")

# ============================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================
print("\n" + "=" * 50)
print("STEP 3: Train-Test Split")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# Display class distribution
print(f"\nTraining class distribution:")
train_dist = pd.Series(y_train).value_counts().sort_index()
for i, count in train_dist.items():
    print(f"  {le_target.classes_[i]}: {count} ({count/len(y_train)*100:.1f}%)")

print(f"\nTesting class distribution:")
test_dist = pd.Series(y_test).value_counts().sort_index()
for i, count in test_dist.items():
    print(f"  {le_target.classes_[i]}: {count} ({count/len(y_test)*100:.1f}%)")

# ============================================================
# STEP 6: SCALE FEATURES
# ============================================================
print("\n" + "=" * 50)
print("STEP 4: Scaling Features")
print("=" * 50)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

# ============================================================
# STEP 7: TRAIN GRADIENT BOOSTING CLASSIFIER
# ============================================================
print("\n" + "=" * 50)
print("STEP 5: Training Gradient Boosting Classifier")
print("=" * 50)

gbc = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

print("Training model...")
gbc.fit(X_train_scaled, y_train)
print("✅ Training complete")

# ============================================================
# STEP 8: MAKE PREDICTIONS
# ============================================================
print("\n" + "=" * 50)
print("STEP 6: Making Predictions")
print("=" * 50)

y_train_pred = gbc.predict(X_train_scaled)
y_test_pred = gbc.predict(X_test_scaled)

# ============================================================
# STEP 9: EVALUATION METRICS
# ============================================================
print("\n" + "=" * 50)
print("STEP 7: Model Evaluation")
print("=" * 50)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n📊 TRAINING SET ACCURACY: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"📊 TEST SET ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ============================================================
# STEP 10: CONFUSION MATRIX
# ============================================================
print("\n" + "=" * 50)
print("STEP 8: Confusion Matrix")
print("=" * 50)

cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print("                 Predicted")
print("              ", "   ".join([f"{le_target.classes_[i]}" for i in range(len(le_target.classes_))]))
print("-" * 50)
for i in range(len(cm)):
    print(f"Actual {le_target.classes_[i]:8} | " + "   ".join([f"{cm[i][j]:4d}" for j in range(len(cm))]))

# ============================================================
# STEP 11: CLASSIFICATION REPORT (FIXED)
# ============================================================
print("\n" + "=" * 50)
print("STEP 9: Classification Report")
print("=" * 50)

# Get unique classes in test set to match target_names
unique_test_classes = np.unique(y_test)
test_class_names = [le_target.classes_[i] for i in unique_test_classes]

print(f"\nClasses present in test set: {test_class_names}")

# Use labels parameter to specify which classes to include
print("\nDetailed Classification Report:")
print(classification_report(
    y_test, 
    y_test_pred, 
    labels=unique_test_classes,  # ← FIX: use only classes present in test set
    target_names=test_class_names,  # ← FIX: match names to present classes
    zero_division=0
))

# ============================================================
# STEP 12: FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 50)
print("STEP 10: Feature Importance")
print("=" * 50)

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': gbc.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance (Top 10):")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")

# ============================================================
# STEP 13: SAMPLE PREDICTIONS
# ============================================================
print("\n" + "=" * 50)
print("STEP 11: Sample Predictions")
print("=" * 50)

print("\n📊 Sample | Actual | Predicted | Correct? | Confidence")
print("-" * 70)

# Get prediction probabilities
y_test_proba = gbc.predict_proba(X_test_scaled)

for i in range(min(10, len(y_test))):
    actual_class = le_target.classes_[y_test[i]]
    pred_class = le_target.classes_[y_test_pred[i]]
    correct = "✅ YES" if y_test[i] == y_test_pred[i] else "❌ NO"
    confidence = max(y_test_proba[i]) * 100
    print(f"  {i+1:6d} | {actual_class:6} | {pred_class:9} | {correct:6} | {confidence:5.1f}%")

# ============================================================
# STEP 14: BUSINESS SUMMARY
# ============================================================
print("\n" + "=" * 50)
print("BUSINESS SUMMARY")
print("=" * 50)

if test_accuracy >= 0.9:
    quality = "EXCELLENT"
elif test_accuracy >= 0.8:
    quality = "GOOD"
elif test_accuracy >= 0.7:
    quality = "MODERATE"
else:
    quality = "POOR"

print(f"""
📊 MODEL PERFORMANCE SUMMARY:

  • Total Samples: {len(df)}
  • Training Samples: {len(X_train)}
  • Testing Samples: {len(X_test)}
  • Features Used: {len(feature_cols)}
  • Target Classes: {list(le_target.classes_)}
  
  • Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
  • Model Quality Rating: {quality}

🎯 MOST IMPORTANT FEATURES:
  1. {feature_importance.iloc[0]['Feature']}: {feature_importance.iloc[0]['Importance']*100:.1f}%
  2. {feature_importance.iloc[1]['Feature']}: {feature_importance.iloc[1]['Importance']*100:.1f}%
  3. {feature_importance.iloc[2]['Feature']}: {feature_importance.iloc[2]['Importance']*100:.1f}%

📋 RECOMMENDATIONS:
  • Focus data collection on top 3 features
  • Retrain model monthly as new data arrives
  • Monitor performance for concept drift
""")

print("=" * 80)
print("✅ GRADIENT BOOSTING CLASSIFIER COMPLETE!")
print("=" * 80)