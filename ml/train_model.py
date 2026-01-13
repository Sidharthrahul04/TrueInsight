import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------- LOAD DATA ----------
df = pd.read_csv("review_dataset.csv")

# Features and label
X = df.drop("label", axis=1)
y = df["label"]

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- TRAIN MODEL ----------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------- EVALUATE ----------
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# ---------- SAVE MODEL ----------
joblib.dump(model, "review_model.pkl")

print("✅ Model trained and saved as review_model.pkl")
