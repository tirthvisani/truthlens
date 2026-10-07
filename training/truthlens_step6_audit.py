import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# TruthLens AI - Step 6
# Dataset leakage and robustness audit
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "truthlens_train.csv"
TEST_PATH = BASE_DIR / "truthlens_test.csv"
MODEL_PATH = BASE_DIR / "truthlens_model.pkl"
VECTORIZER_PATH = BASE_DIR / "truthlens_tfidf_vectorizer.pkl"
REPORT_PATH = BASE_DIR / "truthlens_step6_audit_report.txt"
ERRORS_PATH = BASE_DIR / "truthlens_step6_high_confidence_errors.csv"


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


print("=" * 72)
print("TruthLens AI - Step 6: Dataset Leakage & Robustness Audit")
print("=" * 72)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

for df in (train_df, test_df):
    df["content"] = df["content"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["normalized_content"] = df["content"].map(normalize_text)

print(f"Training rows : {len(train_df):,}")
print(f"Testing rows  : {len(test_df):,}")

# ------------------------------------------------------------
# 1. Duplicate checks
# ------------------------------------------------------------

train_exact_dupes = train_df["content"].duplicated().sum()
test_exact_dupes = test_df["content"].duplicated().sum()

train_norm_dupes = train_df["normalized_content"].duplicated().sum()
test_norm_dupes = test_df["normalized_content"].duplicated().sum()

train_exact_set = set(train_df["content"])
test_exact_set = set(test_df["content"])
exact_overlap = train_exact_set.intersection(test_exact_set)

train_norm_set = set(train_df["normalized_content"])
test_norm_set = set(test_df["normalized_content"])
normalized_overlap = train_norm_set.intersection(test_norm_set)

print("\nDUPLICATE CHECKS")
print("-" * 72)
print(f"Exact duplicates inside train        : {train_exact_dupes:,}")
print(f"Exact duplicates inside test         : {test_exact_dupes:,}")
print(f"Normalized duplicates inside train   : {train_norm_dupes:,}")
print(f"Normalized duplicates inside test    : {test_norm_dupes:,}")
print(f"Exact train/test overlap             : {len(exact_overlap):,}")
print(f"Normalized train/test overlap        : {len(normalized_overlap):,}")

# ------------------------------------------------------------
# 2. Conflicting labels for identical normalized text
# ------------------------------------------------------------

combined = pd.concat(
    [
        train_df[["normalized_content", "label"]].assign(split="train"),
        test_df[["normalized_content", "label"]].assign(split="test"),
    ],
    ignore_index=True,
)

label_counts = combined.groupby("normalized_content")["label"].nunique()
conflicting_texts = label_counts[label_counts > 1]

print("\nLABEL CONSISTENCY")
print("-" * 72)
print(f"Identical normalized texts with conflicting labels: {len(conflicting_texts):,}")

# ------------------------------------------------------------
# 3. Content-length comparison
# ------------------------------------------------------------

train_lengths = train_df["content"].str.len()
test_lengths = test_df["content"].str.len()

print("\nCONTENT LENGTH")
print("-" * 72)
print(f"Train median characters : {train_lengths.median():.0f}")
print(f"Test median characters  : {test_lengths.median():.0f}")
print(f"Train mean characters   : {train_lengths.mean():.1f}")
print(f"Test mean characters    : {test_lengths.mean():.1f}")

# ------------------------------------------------------------
# 4. Re-evaluate current production model
# ------------------------------------------------------------

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

x_test = vectorizer.transform(test_df["content"])
y_test = test_df["label"]

pred = model.predict(x_test)
proba = model.predict_proba(x_test)
confidence = proba.max(axis=1)

cm = confusion_matrix(y_test, pred)

print("\nCURRENT MODEL")
print("-" * 72)
print(classification_report(
    y_test,
    pred,
    target_names=["Real", "Fake"],
    digits=4,
))
print("Confusion Matrix:")
print(cm)

# ------------------------------------------------------------
# 5. High-confidence mistakes
# ------------------------------------------------------------

error_df = test_df.loc[pred != y_test, ["content", "label"]].copy()
error_df["predicted_label"] = pred[pred != y_test]
error_df["confidence"] = confidence[pred != y_test]
error_df["actual"] = error_df["label"].map({0: "Real", 1: "Fake"})
error_df["predicted"] = error_df["predicted_label"].map({0: "Real", 1: "Fake"})

high_conf_errors = error_df[error_df["confidence"] >= 0.90].copy()
high_conf_errors = high_conf_errors.sort_values("confidence", ascending=False)
high_conf_errors.to_csv(ERRORS_PATH, index=False)

print("\nERROR ANALYSIS")
print("-" * 72)
print(f"Total mistakes                 : {len(error_df):,}")
print(f"Mistakes with >=90% confidence : {len(high_conf_errors):,}")

# ------------------------------------------------------------
# 6. Verdict
# ------------------------------------------------------------

issues = []

if len(exact_overlap) > 0:
    issues.append(
        f"Exact train/test leakage detected: {len(exact_overlap):,} overlapping articles."
    )

if len(normalized_overlap) > 0:
    issues.append(
        f"Normalized train/test leakage detected: {len(normalized_overlap):,} overlaps."
    )

if len(conflicting_texts) > 0:
    issues.append(
        f"Conflicting labels detected for {len(conflicting_texts):,} identical texts."
    )

print("\nAUDIT VERDICT")
print("=" * 72)

if issues:
    print("The current evaluation may be optimistic.")
    for issue in issues:
        print("-", issue)
    print("\nRecommended next step: clean/deduplicate the dataset and rebuild the split.")
else:
    print("No exact or normalized train/test leakage was detected.")
    print("The next concern is domain shift: real-world articles may differ from the")
    print("dataset used for training. The next step should be an external challenge set.")

report = [
    "TruthLens AI - Step 6 Audit Report",
    "=" * 72,
    f"Training rows: {len(train_df):,}",
    f"Testing rows: {len(test_df):,}",
    "",
    f"Exact duplicates inside train: {train_exact_dupes:,}",
    f"Exact duplicates inside test: {test_exact_dupes:,}",
    f"Normalized duplicates inside train: {train_norm_dupes:,}",
    f"Normalized duplicates inside test: {test_norm_dupes:,}",
    f"Exact train/test overlap: {len(exact_overlap):,}",
    f"Normalized train/test overlap: {len(normalized_overlap):,}",
    f"Conflicting normalized texts: {len(conflicting_texts):,}",
    "",
    f"Total model mistakes: {len(error_df):,}",
    f"High-confidence mistakes (>=90%): {len(high_conf_errors):,}",
    "",
    "Audit verdict:",
]

if issues:
    report.extend(issues)
else:
    report.append("No exact/normalized leakage found; investigate domain shift next.")

REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

print("\nFiles created:")
print(f"  {REPORT_PATH.name}")
print(f"  {ERRORS_PATH.name}")
print("\nStep 6 finished successfully.")
