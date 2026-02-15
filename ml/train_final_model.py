import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("final_review_dataset.csv")

print("Original label values:", df["label"].unique())

# ================================
# FORCE BINARY LABELS
# ================================
df["label"] = df["label"].apply(lambda x: 1 if x >= 1 else 0)

print("After binarization:", df["label"].unique())

# ================================
# FEATURES
# ================================
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

X = df[FEATURE_COLUMNS]
y = df["label"]

# ================================
# TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================================
# MODEL
# ================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

# ================================
# TRAIN
# ================================
model.fit(X_train, y_train)

# ================================
# EVALUATE
# ================================
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ================================
# SAVE MODEL
# ================================
joblib.dump(
    {"model": model, "features": FEATURE_COLUMNS},
    "../model/review_model.pkl"
)

print("\n✅ Random Forest model trained and saved successfully")
