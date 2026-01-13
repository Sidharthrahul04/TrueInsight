import pandas as pd

df = pd.read_csv("final_review_dataset.csv")

# burst_flag = 1 if many reviews per day, else 0
df["burst_flag"] = (df["daily_review_count"] >= 3).astype(int)

# Reorder columns (IMPORTANT)
df = df[
    [
        "review_length",
        "word_count",
        "sentiment",
        "rating",
        "user_review_count",
        "daily_review_count",
        "duplicate_flag",
        "burst_flag",
        "label"
    ]
]

df.to_csv("final_review_dataset.csv", index=False)
print("✅ Dataset fixed: burst_flag added")
