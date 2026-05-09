import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

# ==============================
# Create output folder
# ==============================

os.makedirs("../outputs/figures", exist_ok=True)

# ==============================
# Load processed dataset
# ==============================

df = pd.read_csv("../data/processed/healthcare_processed.csv")

# ==============================
# Basic Visualizations
# ==============================

# Age Group Distribution
df['age_group'].value_counts().plot(kind='bar')
plt.title("Patient Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../outputs/figures/age_distribution.png")
plt.clf()

# Readmission Rate by Age Group
readmission_rate = df.groupby('age_group')['readmitted_30_days'].mean()

readmission_rate.plot(kind='bar')

plt.title("Readmission Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Rate")
plt.tight_layout()
plt.savefig("../outputs/figures/readmission_by_age.png")
plt.clf()

# High Risk Distribution
df['high_risk_flag'].value_counts().plot(kind='bar')

plt.title("High Risk vs Low Risk Patients")
plt.xlabel("Risk Flag")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../outputs/figures/risk_distribution.png")
plt.clf()

# ==============================
# Machine Learning Evaluation
# ==============================

# Define features and target
X = df.drop(columns=["readmitted_30_days", "patient_id"], errors="ignore")
y = df["readmitted_30_days"]

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load trained model
model = joblib.load("../models/readmission_model.pkl")

# ==============================
# ROC Curve
# ==============================

RocCurveDisplay.from_estimator(model, X_test, y_test)

plt.title("ROC Curve")
plt.savefig("../outputs/figures/roc_curve.png")
plt.clf()

# ==============================
# Confusion Matrix
# ==============================

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

plt.title("Confusion Matrix")
plt.savefig("../outputs/figures/confusion_matrix.png")
plt.clf()

# ==============================
# Feature Importance
# ==============================

importance = model.feature_importances_

feat_importance = pd.Series(
    importance,
    index=X.columns
)

feat_importance.nlargest(10).plot(kind='barh')

plt.title("Top 10 Feature Importances")
plt.tight_layout()

plt.savefig("../outputs/figures/feature_importance.png")
plt.clf()

print("All figures created successfully.")