import pandas as pd

system_df = pd.read_csv("ml/review_dataset.csv")
kaggle_df = pd.read_csv("ml/kaggle_features.csv")

# Add missing columns to kaggle dataset
for col in system_df.columns:
    if col not in kaggle_df.columns:
        kaggle_df[col] = 0

# Reorder columns
kaggle_df = kaggle_df[system_df.columns]

final_df = pd.concat([system_df, kaggle_df], ignore_index=True)
final_df = final_df.sample(frac=1).reset_index(drop=True)

final_df.to_csv("ml/final_review_dataset.csv", index=False)
print("✅ Final dataset fixed and merged correctly")
