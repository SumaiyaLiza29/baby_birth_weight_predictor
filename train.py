import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import clone

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ===============================
# Create models directory
# ===============================
os.makedirs("models", exist_ok=True)

# ===============================
# Load and clean data
# ===============================
print("Loading dataset...")

df = pd.read_csv("data/final_continuous_babies_data.csv")
df = df.drop_duplicates().reset_index(drop=True)

num_cols = ["gestation", "parity", "age", "height", "weight"]

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

df["smoke"] = df["smoke"].fillna(df["smoke"].mode()[0])
df["smoke"] = df["smoke"].astype(str).str.lower().apply(
    lambda x: 1 if ("until" in x or "current" in x or "yes" in x) else 0
)

# ===============================
# BMI Calculation
# ===============================
weight_kg = df["weight"] * 0.453592
height_m  = df["height"] * 0.0254

df["BMI"] = np.where(height_m > 0, weight_kg / (height_m ** 2), np.nan)
df["BMI"] = df["BMI"].fillna(df["BMI"].mean())

# ===============================
# Target variable
# ===============================
df["bwt_class"] = df["bwt"].apply(
    lambda x: 0 if x < 88 else (1 if x <= 141 else 2)
)

X = df[["gestation", "parity", "age", "height", "weight", "BMI", "smoke"]]
y = df["bwt_class"]

print("Dataset prepared.")

# ===============================
# Scaling
# ===============================
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# Handle imbalance
# ===============================
print("Applying SMOTEENN...")

if y.value_counts().min() >= 2:
    X_res, y_res = SMOTEENN(random_state=42).fit_resample(X_scaled, y)
else:
    X_res, y_res = RandomOverSampler(random_state=42).fit_resample(X_scaled, y)

print("Resampling completed.")

# ===============================
# Base models
# ===============================
base_models = [
    ("lgbm", LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )),
    ("xgb", XGBClassifier(
        booster="dart",
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="mlogloss",
        random_state=42
    )),
    ("gb", GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )),
]

# ===============================
# Train models (Stacking)
# ===============================
print("Training models...")

trained_models = []
meta_features = []

for name, model in base_models:
    m = clone(model)
    m.fit(X_res, y_res)

    trained_models.append((name, m))
    meta_features.append(m.predict_proba(X_res))

    print(f"✔ {name} trained")

# ===============================
# Meta model
# ===============================
meta_X = np.hstack(meta_features)

meta_model = LogisticRegression(
    max_iter=2000,
    multi_class="multinomial"
)

meta_model.fit(meta_X, y_res)

print("✔ Meta model trained")

# ===============================
# Save models
# ===============================
print("Saving models...")

for name, m in trained_models:
    joblib.dump(m, f"models/{name}.pkl")

joblib.dump(meta_model, "models/meta_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ All models saved successfully → models/")