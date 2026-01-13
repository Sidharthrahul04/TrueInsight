import mysql.connector
import pandas as pd
from textblob import TextBlob
from collections import Counter
from datetime import datetime

# ---------- DB CONFIG ----------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="trueinsight"
)

cursor = db.cursor(dictionary=True)

cursor.execute("""
SELECT r.id, r.user_id, r.rating, r.review_text, r.created_at
FROM reviews r
""")

reviews = cursor.fetchall()

texts = [r["review_text"].lower().strip() for r in reviews]
text_counts = Counter(texts)

rows = []

for r in reviews:
    text = r["review_text"]

    sentiment = TextBlob(text).sentiment.polarity
    word_count = len(text.split())
    review_length = len(text)

    duplicate_flag = 1 if text_counts[text.lower().strip()] > 1 else 0
    generic_flag = 1 if text.lower().strip() in ["good", "nice", "excellent", "very good"] else 0

    cursor.execute(
        "SELECT COUNT(*) AS cnt FROM reviews WHERE user_id=%s",
        (r["user_id"],)
    )
    user_review_count = cursor.fetchone()["cnt"]

    cursor.execute(
        """
        SELECT COUNT(*) AS cnt FROM reviews
        WHERE user_id=%s AND DATE(created_at)=DATE(%s)
        """,
        (r["user_id"], r["created_at"])
    )
    daily_count = cursor.fetchone()["cnt"]

    # ---------- TEMP LABEL (WEAK SUPERVISION) ----------
    label = 1 if (
        duplicate_flag or
        generic_flag or
        word_count < 5 or
        daily_count >= 3
    ) else 0

    rows.append({
        "review_length": review_length,
        "word_count": word_count,
        "sentiment": sentiment,
        "rating": r["rating"],
        "duplicate_flag": duplicate_flag,
        "generic_flag": generic_flag,
        "user_review_count": user_review_count,
        "daily_review_count": daily_count,
        "label": label
    })

df = pd.DataFrame(rows)
df.to_csv("review_dataset.csv", index=False)

print("✅ Dataset created: review_dataset.csv")
