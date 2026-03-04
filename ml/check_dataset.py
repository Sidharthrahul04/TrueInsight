import pandas as pd

df = pd.read_csv("final_review_dataset.csv")

print("Dataset size:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())