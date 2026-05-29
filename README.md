# 📊 Complete Workshop Guide: 5-Day Data Science & Analytics Program

## 🎯 Workshop Overview

This comprehensive 5-day workshop covers foundational to advanced machine learning concepts, practical business use cases, and hands-on coding exercises. Participants will learn using both **Python** (for flexibility and depth) and **SADSA-Ultimate** (for GUI-based rapid analysis).

| Day | Topic | Key Algorithms | Business Applications |
|:---|:---|:---|:---|
| **Day 1** | Supervised Learning Foundations | Logistic Regression, Decision Trees, Random Forests | Customer churn prediction, credit scoring, risk classification |
| **Day 2** | Classification & Similarity Models | Naive Bayes, KNN, SVM, Neural Networks | Sentiment analysis, fraud detection, quality control |
| **Day 3** | Clustering & Dimensionality Reduction | K-Means, Hierarchical, DBSCAN, PCA | Customer segmentation, anomaly detection |
| **Day 4** | Association Rules & Recommender Systems | Apriori, Collaborative Filtering | Market basket analysis, cross-selling |
| **Day 5** | Ensemble Learning & Model Robustness | XGBoost, Random Forest, Stacking | Campaign response, demand forecasting |

---

## 🛠️ Tool 1: SADSA-Ultimate Installation Guide

[SADSA-Ultimate](https://github.com/codingfigs/SADSA-ULTIMATE) is a comprehensive desktop application for statistical analysis and machine learning without coding. It serves as an excellent companion to Python for quick prototyping and GUI-based exploration.

### 📥 Installation Steps

1. **Download the Installer**
   - Visit the [SADSA-Ultimate Releases Page](https://github.com/codingfigs/SADSA-ULTIMATE/releases/tag/v1.3.0)
   - Download the latest setup executable (v1.3.0)

2. **Run as Administrator**
   - Right-click on the downloaded `.exe` file
   - Select **"Run as administrator"**
   - Click "Yes" when prompted by User Account Control

3. **Follow Installation Wizard**
   - Click "Next" on the welcome screen
   - Accept the License Agreement
   - Choose installation directory (default: `C:\Program Files\SADSA`)
   - Click "Install" (installation takes 1-2 minutes)

4. **Complete Installation**
   - Click "Finish" once installation completes
   - Desktop shortcut is created automatically

### 🚀 First Launch & Trial

- Double-click the SADSA icon on your desktop
- First launch may take 10-15 seconds to initialize
- **30-day full-feature trial** starts automatically – no registration required

### 💻 Using the Python Console in SADSA

SADSA includes an **interactive Python console** that bridges GUI and code:

**Access:** `Help → Python Console` or `Tools → Python Console`

**Key Features:**
- Access your loaded data as `df` (pandas DataFrame)
- Pre-imported libraries: `pd` (pandas), `np` (numpy), `plt` (matplotlib), `sns` (seaborn)
- Execute multi-line code with `Ctrl+Return`
- Call any SADSA analysis function directly (e.g., `app.perform_correlation_analysis()`)

**Example Console Usage:**

```python
# View your data
df.head()
df.describe()

# Create a new computed column
df['Income_per_Year'] = df['AnnualIncome'] / df['Age']

# Call SADSA's built-in visualization
app.generate_plot()  # Opens the plot generation interface