import mysql.connector
import pandas as pd
from textblob import TextBlob
from collections import Counter
from sklearn.metrics import f1_score

# -----------------------------
# DB CONNECTION
# -----------------------------

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="trueinsight"
)

cursor = db.cursor(dictionary=True)

cursor.execute("""
SELECT id, user_id, rating, review_text, created_at
FROM reviews
""")

reviews = cursor.fetchall()

# -----------------------------
# PREPROCESS REVIEWS
# -----------------------------

texts = [r["review_text"].lower().strip() for r in reviews]
text_counts = Counter(texts)

rows = []
scores = []

print("\nExtracting features from system reviews...\n")

for r in reviews:

    text = r["review_text"]
    sentiment = TextBlob(text).sentiment.polarity

    word_count = len(text.split())
    review_length = len(text)

    duplicate_flag = 1 if text.lower().strip() in text_counts and text_counts[text.lower().strip()] > 1 else 0

    generic_flag = 1 if text.lower().strip() in ["good", "nice", "excellent", "very good"] else 0

    # Total reviews by user
    cursor.execute(
        "SELECT COUNT(*) AS cnt FROM reviews WHERE user_id=%s",
        (r["user_id"],)
    )
    user_review_count = cursor.fetchone()["cnt"]

    # Reviews by same user on same day
    cursor.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM reviews
        WHERE user_id=%s AND DATE(created_at)=DATE(%s)
        """,
        (r["user_id"], r["created_at"])
    )
    daily_count = cursor.fetchone()["cnt"]

    # -----------------------------
    # WEAK SUPERVISION SCORE
    # -----------------------------

    score = 0

    if duplicate_flag:
        score += 3

    if daily_count >= 3:
        score += 2

    if generic_flag:
        score += 1

    if word_count < 5:
        score += 1

    if user_review_count > 20:
        score += 1

    if sentiment > 0.8 and r["rating"] == 5:
        score += 1

    scores.append(score)

    rows.append({
        "review_length": review_length,
        "word_count": word_count,
        "sentiment": sentiment,
        "rating": r["rating"],
        "duplicate_flag": duplicate_flag,
        "generic_flag": generic_flag,
        "user_review_count": user_review_count,
        "daily_review_count": daily_count,
        "score": score
    })

df = pd.DataFrame(rows)

# -----------------------------
# LOAD KAGGLE LABELS
# -----------------------------

kaggle_df = pd.read_csv("kaggle_features.csv")

y_true = kaggle_df["label"].apply(lambda x: 1 if x == 1 else 0).values

print("\nWeak supervision threshold tuning\n")

best_threshold = 3
best_f1 = 0

for t in range(1, 6):

    y_pred = [1 if s >= t else 0 for s in scores]

    min_len = min(len(y_true), len(y_pred))

    f1 = f1_score(y_true[:min_len], y_pred[:min_len])

    print(f"Threshold {t} → F1 Score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("\nBest Weak Supervision Threshold:", best_threshold)

# -----------------------------
# FINAL DATASET
# -----------------------------

df["label"] = df["score"].apply(lambda x: 1 if x >= best_threshold else 0)

df.drop(columns=["score"], inplace=True)

df.to_csv("review_dataset.csv", index=False)

print("\nSystem dataset size:", len(df))
print(df["label"].value_counts())

print("\nDataset created successfully → review_dataset.csv")