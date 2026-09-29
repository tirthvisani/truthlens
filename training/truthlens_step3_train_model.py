import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# TruthLens AI - Step 3
# Train the Logistic Regression model
# ============================================================

TRAIN_PATH = "truthlens_train.csv"
TEST_PATH = "truthlens_test.csv"
VECTORIZER_PATH = "truthlens_tfidf_vectorizer.pkl"
MODEL_PATH = "truthlens_model.pkl"

# 1. Load train/test data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train_text = train_df["content"].fillna("").astype(str)
y_train = train_df["label"].astype(int)

X_test_text = test_df["content"].fillna("").astype(str)
y_test = test_df["label"].astype(int)

# 2. Load the TF-IDF vectorizer created in Step 2
tfidf = joblib.load(VECTORIZER_PATH)

# 3. Convert text to TF-IDF features
# Do NOT fit the vectorizer again.
X_train = tfidf.transform(X_train_text)
X_test = tfidf.transform(X_test_text)

print("=" * 60)
print("TruthLens AI - Step 3: Training Model")
print("=" * 60)

print(f"Training samples : {X_train.shape[0]:,}")
print(f"Testing samples  : {X_test.shape[0]:,}")
print(f"Features         : {X_train.shape[1]:,}")

# 4. Create Logistic Regression classifier
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="liblinear",
    random_state=42
)

# 5. Train the model
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)
print("Training completed.")

# 6. Predict the unseen test data
y_pred = model.predict(X_test)

# 7. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

# 8. Save the trained model
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully:")
print(f"  {MODEL_PATH}")
print("\nStep 3 finished successfully.")
