import pandas as pd

# Load datasets
system_df = pd.read_csv("ml/review_dataset.csv")
kaggle_df = pd.read_csv("ml/kaggle_features.csv")

print("System dataset size:", system_df.shape)
print("Kaggle dataset size:", kaggle_df.shape)

# Combine both datasets
final_df = pd.concat([system_df, kaggle_df], ignore_index=True)

# Shuffle dataset
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Save final dataset
final_df.to_csv("ml/final_review_dataset.csv", index=False)

print("✅ Final dataset created: final_review_dataset.csv")
