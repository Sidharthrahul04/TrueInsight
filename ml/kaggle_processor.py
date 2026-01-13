import pandas as pd
import random
from textblob import TextBlob

# Load Kaggle dataset
df = pd.read_csv("ml/kaggle_data/fake_reviews_dataset.csv")

print("Columns found:", df.columns)

# Rename columns if needed (adjust ONLY if error occurs)
# Expected columns: review_text, rating, label

# Feature extraction
features = []

for _, row in df.iterrows():
    text = str(row[0])
    label = int(row[1])  # 1 = fake, 0 = genuine

    polarity = TextBlob(text).sentiment.polarity
    word_count = len(text.split())
    review_length = len(text)

    # Simulated behavioral features
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

# Create final dataframe
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

# Save processed dataset
final_df.to_csv("ml/kaggle_features.csv", index=False)

print("✅ Kaggle dataset processed and saved as kaggle_features.csv")
