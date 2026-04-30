import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

n = 50000

data = pd.DataFrame({
    "patient_id": range(1, n + 1),
    "age": np.random.randint(18, 90, n),
    "length_of_stay": np.random.randint(1, 15, n),
    "num_prior_visits": np.random.randint(0, 10, n),
    "num_medications": np.random.randint(1, 25, n),
    "has_diabetes": np.random.choice([0, 1], n, p=[0.75, 0.25]),
    "has_hypertension": np.random.choice([0, 1], n, p=[0.65, 0.35]),
    "discharge_type": np.random.choice(
        ["Home", "Skilled Nursing", "Rehab", "Against Medical Advice"], n
    ),
    "insurance_type": np.random.choice(
        ["Private", "Medicare", "Medicaid", "Self-Pay"], n
    ),
})

risk_score = (
    0.03 * data["age"]
    + 0.25 * data["length_of_stay"]
    + 0.40 * data["num_prior_visits"]
    + 0.15 * data["num_medications"]
    + 1.2 * data["has_diabetes"]
    + 1.0 * data["has_hypertension"]
)

probability = 1 / (1 + np.exp(-(risk_score - 8)))
data["readmitted_30_days"] = np.random.binomial(1, probability)

Path("data/raw").mkdir(parents=True, exist_ok=True)
data.to_csv("data/raw/healthcare_readmission.csv", index=False)

print("Dataset created successfully.")
print(data.head())
print(data.shape)