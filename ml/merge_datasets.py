import pandas as pd

# -----------------------------

# LOAD DATASETS

# -----------------------------

system_df = pd.read_csv("review_dataset.csv")
kaggle_df = pd.read_csv("kaggle_features.csv")

print("System dataset:", len(system_df))
print("Kaggle dataset:", len(kaggle_df))

# -----------------------------

# ENSURE LABELS ARE BINARY

# -----------------------------

system_df["label"] = system_df["label"].apply(lambda x: 1 if x == 1 else 0)
kaggle_df["label"] = kaggle_df["label"].apply(lambda x: 1 if x == 1 else 0)

# -----------------------------

# MATCH COLUMN STRUCTURE

# -----------------------------

for col in system_df.columns:
	if col not in kaggle_df.columns:
		kaggle_df[col] = 0

# Ensure same column order

kaggle_df = kaggle_df[system_df.columns]

# -----------------------------

# MERGE DATASETS

# -----------------------------

merged = pd.concat([system_df, kaggle_df], ignore_index=True)

print("\nMerged distribution:")
print(merged["label"].value_counts())

# -----------------------------

# BALANCE THE DATASET

# -----------------------------

fake = merged[merged["label"] == 1]
genuine = merged[merged["label"] == 0]

min_size = min(len(fake), len(genuine))

fake = fake.sample(min_size, random_state=42)
genuine = genuine.sample(min_size, random_state=42)

balanced_df = pd.concat([fake, genuine])

# Shuffle rows

balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced distribution:")
print(balanced_df["label"].value_counts())

# -----------------------------

# SAVE FINAL DATASET

# -----------------------------

balanced_df.to_csv("final_review_dataset.csv", index=False)

print("\n✅ Balanced dataset saved successfully")
