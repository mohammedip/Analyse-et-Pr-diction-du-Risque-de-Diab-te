# 🧪 Diabetes Risk Prediction & Clustering Project

## 📌 Project Overview

This project aims to develop an intelligent system capable of predicting whether a patient is at high risk of developing diabetes. Using historical clinical data, the system performs both **unsupervised clustering** to identify patient profiles and **supervised classification** to predict diabetes risk.

The clinical features used include:

* **Glucose** (Glycemia)
* **Blood Pressure**
* **Skin Thickness**
* **Insulin Levels**
* **BMI** (Body Mass Index)
* **Diabetes Pedigree Function** (Genetic predisposition)
* **Age**

---

## 🎯 Objectives

* Perform **Exploratory Data Analysis (EDA)** to understand the dataset.
* Clean and preprocess the data (missing values, outliers, scaling).
* Apply **K-Means Clustering** to discover patient groups.
* Detect **high-risk diabetes clusters**.
* Build **supervised classification models** to predict diabetes risk.
* Evaluate, compare and save the best-performing model.

---

## 🗂️ Project Structure

```
📁 diabetes-risk-project
├── 📁 data/              # Raw and processed datasets
├── 📁 notebooks/         # Jupyter/Colab notebooks for analysis
├── 📁 models/            # Saved trained models (e.g., model.pkl)
├── 📁 src/               # Python scripts for preprocessing and ML
├── README.md             # Project documentation
└── requirements.txt      # Required libraries
```

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

Make sure you have Python 3.8+ installed.

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Project

```bash
# Example: Launch the notebook
jupyter notebook notebooks/EDA.ipynb
```

---

## 📊 User Story Summary

### 🧭 **User Story 1: Data Exploration (EDA)**

* Load data with `pandas`
* Analyze shape, types, missing values, duplicates
* Plot distributions & correlations

### 🧼 **User Story 2: Data Preprocessing**

* Handle missing values & zeros
* Detect and treat outliers (IQR, Z-score)
* Normalize/standardize features

### 🎨 **User Story 3: Clustering with K-Means**

* Choose optimal `k` (Elbow & Silhouette)
* Train K-Means and assign cluster labels
* Visualize clusters

### 🧠 **User Story 4: Cluster Analysis**

* Compute cluster means
* Identify **high-risk cluster** (Glucose >126, BMI >30, DPF >0.5)
* Create `risk_category` column

### 🤖 **User Story 5: Supervised Classification**

* Define **X** (features) & **y** (clusters or risk)
* Train models: RF, SVM, XGB, Logistic Regression...
* Evaluate using Confusion Matrix, Precision, Recall, F1-score
* Use **Cross Validation & Hyperparameter Tuning**
* Save best model (`model.pkl`)

### 📝 **User Story 6: Documentation & Reproducibility**

* Comment code clearly
* Create README.md & Jira tasks

---

## 🧪 Bonus: Streamlit App (Optional)

Create a user interface for real-time diabetes risk prediction:

```bash
streamlit run app.py
```

---

## 🛠️ Technologies Used

| Category       | Tools/Libs                 |
| -------------- | -------------------------- |
| Data Analysis  | Pandas, NumPy              |
| Visualization  | Matplotlib, Seaborn        |
| Clustering     | K-Means (Scikit-learn)     |
| Classification | RandomForest, SVM, XGBoost |
| Deployment     | Streamlit, Pickle          |

---

## 📈 Evaluation Metrics

* **Accuracy**
* **Precision / Recall**
* **F1-Score**
* **Confusion Matrix**
* **Cross-validation**

---

**Ready to predict and prevent diabetes risks with AI 🧠⚕️!**
