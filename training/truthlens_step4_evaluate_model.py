import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

TEST_PATH = "truthlens_test.csv"
VECTORIZER_PATH = "truthlens_tfidf_vectorizer.pkl"
MODEL_PATH = "truthlens_model.pkl"

test_df = pd.read_csv(TEST_PATH)
tfidf = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

X_test_text = test_df["content"].fillna("").astype(str)
y_test = test_df["label"].astype(int)

X_test = tfidf.transform(X_test_text)
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("=" * 65)
print("TruthLens AI - Step 4: Detailed Evaluation")
print("=" * 65)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"], digits=4))

cm = confusion_matrix(y_test, y_pred)

print("=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)
print("Rows = Actual, Columns = Predicted")
print("                 Predicted")
print("                 Real    Fake")
print(f"Actual Real      {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"Actual Fake      {cm[1,0]:5d}   {cm[1,1]:5d}")

plt.figure(figsize=(7, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("TruthLens AI - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["Real", "Fake"])
plt.yticks([0, 1], ["Real", "Fake"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("truthlens_confusion_matrix.png", dpi=150)
plt.close()

results = test_df.copy()
results["actual_label"] = y_test
results["predicted_label"] = y_pred
results["prediction"] = results["predicted_label"].map({0: "Real", 1: "Fake"})
results["confidence"] = y_probability.max(axis=1)
results.to_csv("truthlens_test_predictions.csv", index=False)

errors = results[results["actual_label"] != results["predicted_label"]]

print("\n" + "=" * 65)
print(f"Incorrect predictions: {len(errors):,} / {len(results):,}")
print("=" * 65)

if len(errors) > 0:
    print("\nFirst 5 incorrect predictions:")
    for _, row in errors.head(5).iterrows():
        print("-" * 65)
        print("Actual    :", "Fake" if row["actual_label"] == 1 else "Real")
        print("Predicted :", "Fake" if row["predicted_label"] == 1 else "Real")
        print("Confidence:", f"{row['confidence'] * 100:.2f}%")
        print("Title     :", row.get("title", "Not available"))

print("\nFiles created:")
print("  truthlens_confusion_matrix.png")
print("  truthlens_test_predictions.csv")
print("\nStep 4 finished successfully.")
