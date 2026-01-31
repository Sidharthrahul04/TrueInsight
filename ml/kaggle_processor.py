import pandas as pd
import random
from textblob import TextBlob

df = pd.read_csv("ml/kaggle_data/fake_reviews_dataset.csv")

features = []

for _, row in df.iterrows():
    text = str(row[0])

    # FORCE BINARY LABEL
    raw_label = int(row[1])
    label = 1 if raw_label != 0 else 0   # fake = 1, genuine = 0

    polarity = TextBlob(text).sentiment.polarity
    word_count = len(text.split())
    review_length = len(text)

    user_review_count = random.randint(1, 20)
    daily_review_count = random.randint(1, 5)

    features.append([
        review_length,
        word_count,
        polarity,
        user_review_count,
        daily_review_count,
        label
    ])

final_df = pd.DataFrame(
    features,
    columns=[
        "review_length",
        "word_count",
        "sentiment",
        "user_review_count",
        "daily_review_count",
        "label"
    ]
)

final_df.to_csv("ml/kaggle_features.csv", index=False)
print("✅ Kaggle dataset converted to BINARY labels")
