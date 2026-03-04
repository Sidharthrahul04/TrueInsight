import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import joblib

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("final_review_dataset.csv")

print("Original label values:", df["label"].unique())

# convert to binary
df["label"] = df["label"].astype(int)

print("After binarization:", df["label"].unique())

# =====================================
# FEATURES
# =====================================

FEATURE_COLUMNS = [
    "review_length",
    "word_count",
    "sentiment",
    "rating",
    "user_review_count",
    "daily_review_count",
    "duplicate_flag",
    "generic_flag"
]

X = df[FEATURE_COLUMNS]
y = df["label"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# RANDOM FOREST BASE MODEL
# =====================================

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

# =====================================
# HYPERPARAMETER GRID
# =====================================

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 12, 16],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# =====================================
# GRID SEARCH
# =====================================

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters Found:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# =====================================
# PREDICT PROBABILITIES
# =====================================

y_probs = best_model.predict_proba(X_test)[:, 1]

# =====================================
# THRESHOLD TUNING
# =====================================

thresholds = np.arange(0.30, 0.90, 0.05)

best_threshold = 0
best_f1 = 0

print("\nThreshold tuning results:\n")

for t in thresholds:

    y_pred = (y_probs >= t).astype(int)

    f1 = f1_score(y_test, y_pred)

    print(f"Threshold {t:.2f} → F1 Score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("\nBest Threshold:", best_threshold)

# =====================================
# FINAL PREDICTIONS WITH BEST THRESHOLD
# =====================================

final_pred = (y_probs >= best_threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, final_pred, zero_division=0))

# =====================================
# SAVE MODEL + THRESHOLD
# =====================================

joblib.dump(
    {
        "model": best_model,
        "features": FEATURE_COLUMNS,
        "threshold": best_threshold
    },
    "../model/review_model.pkl"
)

print("\nModel trained, tuned, and saved successfully.")