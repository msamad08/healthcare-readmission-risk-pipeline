import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Load processed data
df = pd.read_csv("data/processed/healthcare_processed.csv")

# One-hot encode categorical columns
df_model = pd.get_dummies(
    df,
    columns=["discharge_type", "insurance_type", "age_group"],
    drop_first=True
)

# Features and target
X = df_model.drop(columns=["patient_id", "readmitted_30_days"])
y = df_model["readmitted_30_days"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=8,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(round(roc_auc_score(y_test, y_prob), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/readmission_risk_model.joblib")

print("\nModel saved to models/readmission_risk_model.joblib")