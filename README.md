\# 📊 Complete Workshop Guide: 5-Day Data Science \& Analytics Program



\## 🎯 Workshop Overview



This comprehensive 5-day workshop covers foundational to advanced machine learning concepts, practical business use cases, and hands-on coding exercises. Participants will learn using both \*\*Python\*\* (for flexibility and depth) and \*\*SADSA-Ultimate\*\* (for GUI-based rapid analysis).



| Day | Topic | Key Algorithms | Business Applications |

|:---|:---|:---|:---|

| \*\*Day 1\*\* | Supervised Learning Foundations | Logistic Regression, Decision Trees, Random Forests | Customer churn prediction, credit scoring, risk classification |

| \*\*Day 2\*\* | Classification \& Similarity Models | Naive Bayes, KNN, SVM, Neural Networks | Sentiment analysis, fraud detection, quality control |

| \*\*Day 3\*\* | Clustering \& Dimensionality Reduction | K-Means, Hierarchical, DBSCAN, PCA | Customer segmentation, anomaly detection |

| \*\*Day 4\*\* | Association Rules \& Recommender Systems | Apriori, Collaborative Filtering | Market basket analysis, cross-selling |

| \*\*Day 5\*\* | Ensemble Learning \& Model Robustness | XGBoost, Random Forest, Stacking | Campaign response, demand forecasting |



\---



\## 🛠️ Tool 1: SADSA-Ultimate Installation Guide



\[SADSA-Ultimate](https://github.com/codingfigs/SADSA-ULTIMATE) is a comprehensive desktop application for statistical analysis and machine learning without coding. It serves as an excellent companion to Python for quick prototyping and GUI-based exploration.



\### 📥 Installation Steps



1\. \*\*Download the Installer\*\*

&#x20;  - Visit the \[SADSA-Ultimate Releases Page](https://github.com/codingfigs/SADSA-ULTIMATE/releases/download/v1.0.2/SADSA-Ultimate-v1.0.2-Setup.exe)

&#x20;  - Download `SADSA-Ultimate-v1.0.2-Setup.exe`



2\. \*\*Run as Administrator\*\*

&#x20;  - Right-click on the downloaded `.exe` file

&#x20;  - Select \*\*"Run as administrator"\*\*

&#x20;  - Click "Yes" when prompted by User Account Control



3\. \*\*Follow Installation Wizard\*\*

&#x20;  - Click "Next" on the welcome screen

&#x20;  - Accept the License Agreement

&#x20;  - Choose installation directory (default: `C:\\Program Files\\SADSA`)

&#x20;  - Click "Install" (installation takes 1-2 minutes)



4\. \*\*Complete Installation\*\*

&#x20;  - Click "Finish" once installation completes

&#x20;  - Desktop shortcut is created automatically



\### 🚀 First Launch \& Trial



\- Double-click the SADSA icon on your desktop

\- First launch may take 10-15 seconds to initialize

\- \*\*30-day full-feature trial\*\* starts automatically – no registration required!



\### 💻 Using the Python Console in SADSA



SADSA includes an \*\*interactive Python console\*\* that bridges GUI and code:



\*\*Access:\*\* `Help → Python Console` or `Tools → Python Console`



\*\*Key Features:\*\*

\- Access your loaded data as `df` (pandas DataFrame)

\- Pre-imported libraries: `pd` (pandas), `np` (numpy), `plt` (matplotlib), `sns` (seaborn)

\- Execute multi-line code with `Ctrl+Return`

\- Call any SADSA analysis function directly (e.g., `app.perform\_correlation\_analysis()`)



\*\*Example Console Usage:\*\*

```python

\# View your data

df.head()

df.describe()



\# Create a new computed column

df\['Income\_per\_Year'] = df\['AnnualIncome'] / df\['Age']



\# Call SADSA's built-in visualization

app.generate\_plot()  # Opens the plot generation interface

```



\---



\## 🐍 Tool 2: Standalone Python Setup (Alternative/Complement)



For participants who prefer coding or need advanced customization:



```bash

\# Create virtual environment

python -m venv sadsa\_env

sadsa\_env\\Scripts\\activate  # Windows



\# Install required packages

pip install pandas numpy matplotlib seaborn scikit-learn

pip install xgboost lightgbm mlxtend

```



\---



\## 📁 Day-by-Day Learning Materials \& Datasets



\### Day 1: Supervised Learning Foundations

\*\*Topics:\*\* Logistic Regression, Decision Trees, Random Forests



| Dataset | File Name | Description |

|:---|:---|:---|

| Customer Churn | `churn\_data.csv` | Bank customer data with churn labels |

| Credit Risk | `credit\_risk.csv` | Loan default prediction dataset |

| Order Delays | `order\_delays.csv` | E-commerce fulfillment delays |



\*\*Python Practice:\*\* `day1\_supervised\_learning.py`

\*\*SADSA Practice:\*\* `Machine Learning → Supervised Learning → \[Logistic Regression, Decision Tree, Random Forest]`



\---



\### Day 2: Classification \& Similarity Models

\*\*Topics:\*\* Naive Bayes, KNN, SVM, Neural Networks



| Dataset | File Name | Description |

|:---|:---|:---|

| Sentiment Reviews | `sentiment\_reviews.csv` | 50,000 product reviews with sentiment |

| Fraud Transactions | `fraud\_transactions.csv` | 200,000 transactions (3.5% fraud) |

| Loan Risk | `loan\_risk\_data.csv` | Multi-class risk classification |



\*\*Python Practice:\*\* `day2\_classification\_models.py`

\*\*SADSA Practice:\*\* `Machine Learning → Supervised Learning → \[Naive Bayes, KNN, SVM, Neural Network]`



\---



\### Day 3: Clustering \& Dimensionality Reduction

\*\*Topics:\*\* K-Means, Hierarchical, DBSCAN, PCA, Factor Analysis



| Dataset | File Name | Description |

|:---|:---|:---|

| Customer Segments | `customer\_segmentation.csv` | 10,000 customers with spending patterns |

| Brand Positioning | `brand\_positioning.csv` | 24 brands with 6 attributes |

| Anomaly Detection | `credit\_card\_transactions.csv` | 28 anonymized features + amount |



\*\*Python Practice:\*\* `day3\_clustering\_pca.py`

\*\*SADSA Practice:\*\* `Machine Learning → Unsupervised Learning → \[K-Means, Hierarchical, DBSCAN, PCA]`



\---



\### Day 4: Association Rules \& Recommender Systems

\*\*Topics:\*\* Apriori Algorithm, Collaborative Filtering, Content-Based Filtering



| Dataset | File Name | Description |

|:---|:---|:---|

| Market Basket | `grocery\_transactions.csv` | 100 transactions, 24 products |

| Credit Card Offers | `customer\_credit\_preferences.csv` | Customer profiles for card recommendations |

| Movie Ratings | `user\_movie\_ratings.csv` | 50 users × 10 movies rating matrix |



\*\*Python Practice:\*\* `day4\_association\_rules.py`, `day4\_recommender.py`

\*\*SADSA Practice:\*\* `Machine Learning → Unsupervised Learning → Association Rules / Recommender System`



\---



\### Day 5: Ensemble Learning \& Model Robustness

\*\*Topics:\*\* Bagging, Boosting (XGBoost, LightGBM), Stacking, Stress Testing



| Dataset | File Name | Description |

|:---|:---|:---|

| Campaign Response | `campaign\_data.csv` | 1,100 customers with response labels |

| Demand Forecasting | `demand\_data.csv` | 250 days of product demand |

| Credit Risk Stress | `credit\_risk\_data.csv` | 500 loan applications for stress testing |



\*\*Python Practice:\*\* `day5\_ensemble\_models.py`, `day5\_stress\_testing.py`

\*\*SADSA Practice:\*\* `Machine Learning → Ensemble Methods → \[XGBoost, LightGBM, Voting, Stacking]`



\---



\## 🚀 Running the Python Demos



Each day’s Python code is self-contained. Simply:



```bash

\# Example for Day 1

python day1\_supervised\_learning.py



\# Example for Day 5

python day5\_ensemble\_models.py

```



Each script will:

1\. Load the corresponding dataset

2\. Train multiple models

3\. Display performance metrics

4\. Generate visualizations (saved as PNG files)

5\. Output business recommendations



\---



\## 🔄 Integrating SADSA-Ultimate with Python Workflow



\### Option 1: Use SADSA for Quick Exploration, Python for Production

1\. \*\*Load data\*\* in SADSA (`File → Open Data File`)

2\. \*\*Explore\*\* using built-in visualizations (`Plots` menu)

3\. \*\*Export\*\* preprocessed data (`File → Save Data File`)

4\. \*\*Build final model\*\* in Python using the exported data



\### Option 2: Use SADSA's Python Console for Hybrid Work

```python

\# Inside SADSA's Python Console:



\# 1. Load data (already in df)

print(f"Data shape: {df.shape}")



\# 2. Perform quick preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled\_data = scaler.fit\_transform(df.select\_dtypes(include=\['float64', 'int64']))



\# 3. Call SADSA's built-in analysis on the result

\# (First, you might need to put the result back into SADSA's context)

\# For clustering, use: app.perform\_kmeans\_ml()

```



\### Option 3: Use SADSA's AI Insights for Interpretation

After running any analysis in SADSA (e.g., from `Machine Learning` menu):

\- Click the \*\*"🤖 AI Insights"\*\* button in the results window

\- The AI will explain the statistical output in plain business language

\- Use this to validate or interpret your Python model results



\---



\## 📊 Sample Python Code Structure (Day 1 Example)



```python

\# day1\_supervised\_learning.py

import pandas as pd

import numpy as np

from sklearn.model\_selection import train\_test\_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification\_report



\# Load data

df = pd.read\_csv('churn\_data.csv')



\# Prepare features

X = df.drop('Churn', axis=1)

y = df\['Churn']



\# Train/test split

X\_train, X\_test, y\_train, y\_test = train\_test\_split(

&#x20;   X, y, test\_size=0.2, random\_state=42

)



\# Train model

rf = RandomForestClassifier(n\_estimators=100)

rf.fit(X\_train, y\_train)



\# Evaluate

y\_pred = rf.predict(X\_test)

print(classification\_report(y\_test, y\_pred))



\# Feature importance

importance = pd.DataFrame({

&#x20;   'feature': X.columns,

&#x20;   'importance': rf.feature\_importances\_

}).sort\_values('importance', ascending=False)

print("\\nTop 5 features:\\n", importance.head())

```



\---



\## 📈 Expected Learning Outcomes



By the end of this workshop, participants will be able to:



| Skill | Application |

|:---|:---|

| \*\*Select appropriate ML algorithms\*\* | Match business problems to solution approaches |

| \*\*Build and evaluate models\*\* | Use Python and SADSA for end-to-end analysis |

| \*\*Interpret model outputs\*\* | Translate technical metrics into business insights |

| \*\*Create recommendation systems\*\* | Implement market basket analysis and collaborative filtering |

| \*\*Ensure model robustness\*\* | Apply stress testing and ensemble methods for reliability |

| \*\*Leverage GUI tools\*\* | Use SADSA for rapid prototyping and AI-assisted insights |



\---



\## 📞 Support \& Resources



\- \*\*SADSA Website:\*\* \[http://codingfigs.com](http://codingfigs.com)

\- \*\*SADSA Support Email:\*\* contact@codingfigs.com

\- \*\*Python Documentation:\*\* \[https://docs.python.org/3/](https://docs.python.org/3/)

\- \*\*Scikit-learn Documentation:\*\* \[https://scikit-learn.org/](https://scikit-learn.org/)



For SADSA license activation or technical issues, contact the developer directly with your \*\*Machine ID\*\* (found under `Help → License Information`).



\---



\## 📝 License \& Usage Notes



\- \*\*SADSA-Ultimate:\*\* Full-featured 30-day trial. Extended use requires purchase of a license.

\- \*\*Python Code:\*\* Free to use, modify, and distribute for educational purposes.

\- \*\*Datasets:\*\* Included for workshop use only; some are derived from public sources.



\---



\*\*Happy Learning! 🚀\*\*

