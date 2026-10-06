import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# TruthLens AI - Step 5
# Improve the text classifier and compare it with the baseline
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "truthlens_train.csv"
TEST_PATH = BASE_DIR / "truthlens_test.csv"

CURRENT_VECTORIZER_PATH = BASE_DIR / "truthlens_tfidf_vectorizer.pkl"
CURRENT_MODEL_PATH = BASE_DIR / "truthlens_model.pkl"

BACKUP_VECTORIZER_PATH = BASE_DIR / "truthlens_tfidf_vectorizer_v1.pkl"
BACKUP_MODEL_PATH = BASE_DIR / "truthlens_model_v1.pkl"

REPORT_PATH = BASE_DIR / "truthlens_model_v2_report.txt"
PREDICTIONS_PATH = BASE_DIR / "truthlens_v2_test_predictions.csv"


def evaluate(name, model, vectorizer, x_test_text, y_test):
    x_test = vectorizer.transform(x_test_text)
    prediction = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, zero_division=0)
    recall = recall_score(y_test, prediction, zero_division=0)
    f1 = f1_score(y_test, prediction, zero_division=0)

    cm = confusion_matrix(y_test, prediction)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Accuracy       : {accuracy * 100:.2f}%")
    print(f"Fake Precision : {precision * 100:.2f}%")
    print(f"Fake Recall    : {recall * 100:.2f}%")
    print(f"Fake F1 Score  : {f1 * 100:.2f}%")
    print("\nConfusion Matrix")
    print("Rows = Actual, Columns = Predicted")
    print("                 Real    Fake")
    print(f"Actual Real      {cm[0, 0]:5d}   {cm[0, 1]:5d}")
    print(f"Actual Fake      {cm[1, 0]:5d}   {cm[1, 1]:5d}")

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prediction": prediction,
        "probabilities": probabilities,
        "cm": cm,
    }


print("=" * 70)
print("TruthLens AI - Step 5: Model Improvement")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

x_train_text = train_df["content"].fillna("").astype(str)
y_train = train_df["label"].astype(int)

x_test_text = test_df["content"].fillna("").astype(str)
y_test = test_df["label"].astype(int)

print(f"Training samples : {len(train_df):,}")
print(f"Testing samples  : {len(test_df):,}")

# ------------------------------------------------------------
# Baseline model currently used by Flask
# ------------------------------------------------------------

baseline_vectorizer = joblib.load(CURRENT_VECTORIZER_PATH)
baseline_model = joblib.load(CURRENT_MODEL_PATH)

baseline_result = evaluate(
    "V1 BASELINE - Word TF-IDF + Logistic Regression",
    baseline_model,
    baseline_vectorizer,
    x_test_text,
    y_test,
)

# ------------------------------------------------------------
# V2 feature extractor
#
# Word features capture phrases and vocabulary.
# Character features help detect spelling patterns, punctuation,
# word fragments, and writing style that word-only TF-IDF misses.
# ------------------------------------------------------------

v2_vectorizer = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=120000,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=80000,
            sublinear_tf=True,
        ),
    ),
])

print("\nBuilding V2 TF-IDF features...")
x_train_v2 = v2_vectorizer.fit_transform(x_train_text)
print(f"V2 feature matrix: {x_train_v2.shape}")

v2_model = LogisticRegression(
    C=2.0,
    max_iter=2000,
    solver="liblinear",
    class_weight="balanced",
    random_state=42,
)

print("Training V2 Logistic Regression...")
v2_model.fit(x_train_v2, y_train)

v2_result = evaluate(
    "V2 - Word + Character TF-IDF + Balanced Logistic Regression",
    v2_model,
    v2_vectorizer,
    x_test_text,
    y_test,
)

# ------------------------------------------------------------
# Save a detailed report
# ------------------------------------------------------------

report_lines = [
    "TruthLens AI - Model V2 Comparison",
    "=" * 70,
    "",
    "V1 BASELINE",
    f"Accuracy: {baseline_result['accuracy'] * 100:.2f}%",
    f"Fake Precision: {baseline_result['precision'] * 100:.2f}%",
    f"Fake Recall: {baseline_result['recall'] * 100:.2f}%",
    f"Fake F1: {baseline_result['f1'] * 100:.2f}%",
    "",
    "V2 ENHANCED",
    f"Accuracy: {v2_result['accuracy'] * 100:.2f}%",
    f"Fake Precision: {v2_result['precision'] * 100:.2f}%",
    f"Fake Recall: {v2_result['recall'] * 100:.2f}%",
    f"Fake F1: {v2_result['f1'] * 100:.2f}%",
    "",
    classification_report(
        y_test,
        v2_result["prediction"],
        target_names=["Real", "Fake"],
        digits=4,
    ),
]

REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

results = test_df.copy()
results["actual_label"] = y_test
results["predicted_label"] = v2_result["prediction"]
results["prediction"] = results["predicted_label"].map({0: "Real", 1: "Fake"})
results["confidence"] = v2_result["probabilities"].max(axis=1)
results.to_csv(PREDICTIONS_PATH, index=False)

# ------------------------------------------------------------
# Promotion rule
#
# Fake-class F1 is the primary metric because earlier manual
# testing exposed fake articles being classified as real.
# Accuracy is used as a guard so we do not improve recall by
# badly damaging overall performance.
# ------------------------------------------------------------

f1_improved = v2_result["f1"] > baseline_result["f1"]
accuracy_acceptable = (
    v2_result["accuracy"] >= baseline_result["accuracy"] - 0.005
)

print("\n" + "=" * 70)
print("PROMOTION DECISION")
print("=" * 70)

if f1_improved and accuracy_acceptable:
    print("V2 PASSED. Promoting V2 to the Flask application.")

    if not BACKUP_VECTORIZER_PATH.exists():
        shutil.copy2(CURRENT_VECTORIZER_PATH, BACKUP_VECTORIZER_PATH)

    if not BACKUP_MODEL_PATH.exists():
        shutil.copy2(CURRENT_MODEL_PATH, BACKUP_MODEL_PATH)

    joblib.dump(v2_vectorizer, CURRENT_VECTORIZER_PATH)
    joblib.dump(v2_model, CURRENT_MODEL_PATH)

    print("\nProduction files updated:")
    print(f"  {CURRENT_VECTORIZER_PATH.name}")
    print(f"  {CURRENT_MODEL_PATH.name}")
    print("\nV1 backups:")
    print(f"  {BACKUP_VECTORIZER_PATH.name}")
    print(f"  {BACKUP_MODEL_PATH.name}")
else:
    print("V2 was NOT promoted.")
    print("The existing V1 production model remains unchanged.")

    if not f1_improved:
        print("- V2 Fake-class F1 did not beat V1.")

    if not accuracy_acceptable:
        print("- V2 accuracy dropped by more than 0.5 percentage points.")

print("\nFiles created:")
print(f"  {REPORT_PATH.name}")
print(f"  {PREDICTIONS_PATH.name}")
print("\nStep 5 finished successfully.")
