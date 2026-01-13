import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

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

X = df[FEATURE_COLUMNS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump(
    {"model": model, "features": FEATURE_COLUMNS},
    "../model/review_model.pkl"
)

print("✅ Model saved correctly with feature metadata")
