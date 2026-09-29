import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ============================================================
# TruthLens AI - Step 2
# Prepare dataset and create TF-IDF features
# ============================================================

# 1. Load cleaned dataset
DATASET_PATH = "truthlens_cleaned_news.csv"

df = pd.read_csv(DATASET_PATH)

# 2. Basic safety checks
df["content"] = df["content"].fillna("").astype(str)
df["label"] = df["label"].astype(int)

# 3. Split into training and testing data
# Stratify keeps the Fake/Real ratio similar in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# 4. Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=100000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)

# 5. Fit ONLY on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# 6. Transform test data using the fitted vectorizer
X_test_tfidf = tfidf.transform(X_test)

# 7. Save the vectorizer for later Flask integration
joblib.dump(tfidf, "truthlens_tfidf_vectorizer.pkl")

# 8. Save the exact train/test splits
train_df = pd.DataFrame({
    "content": X_train,
    "label": y_train
}).reset_index(drop=True)

test_df = pd.DataFrame({
    "content": X_test,
    "label": y_test
}).reset_index(drop=True)

train_df.to_csv("truthlens_train.csv", index=False)
test_df.to_csv("truthlens_test.csv", index=False)

# 9. Display results
print("=" * 55)
print("TruthLens AI - Step 2 Completed")
print("=" * 55)

print(f"Total articles        : {len(df):,}")
print(f"Training articles     : {len(X_train):,}")
print(f"Testing articles      : {len(X_test):,}")

print("\nTraining labels:")
print(f"  Real (0)             : {(y_train == 0).sum():,}")
print(f"  Fake (1)             : {(y_train == 1).sum():,}")

print("\nTesting labels:")
print(f"  Real (0)             : {(y_test == 0).sum():,}")
print(f"  Fake (1)             : {(y_test == 1).sum():,}")

print("\nTF-IDF:")
print(f"  Training shape       : {X_train_tfidf.shape}")
print(f"  Testing shape        : {X_test_tfidf.shape}")

print("\nFiles created:")
print("  truthlens_tfidf_vectorizer.pkl")
print("  truthlens_train.csv")
print("  truthlens_test.csv")

print("\nStep 2 finished successfully.")
