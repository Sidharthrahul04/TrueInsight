import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

# ---------------- LOAD DATA ----------------
df = pd.read_csv("final_review_dataset.csv")

FEATURE_COLUMNS = [
    "review_length",
    "word_count",
    "sentiment",
    "rating",
    "user_review_count",
    "daily_review_count",
    "duplicate_flag",
    "burst_flag"
]

# ---------------- FORCE BINARY LABEL ----------------
df["label"] = df["label"].astype(int)

print("\nLabel distribution:")
print(df["label"].value_counts())

X = df[FEATURE_COLUMNS]
y = df["label"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- PIPELINE ----------------
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced"
    ))
])

# ---------------- TRAIN ----------------
pipeline.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = pipeline.predict(X_test)

print("\nLOGISTIC REGRESSION RESULTS")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------------- SAVE MODEL ----------------
os.makedirs("../model", exist_ok=True)

joblib.dump(
    {
        "model": pipeline,
        "features": FEATURE_COLUMNS
    },
    "../model/review_model_lr.pkl"
)

print("\n✅ Logistic Regression model saved successfully")
