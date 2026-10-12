
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the Kaggle dataset
DATA_PATH = "data/phishing_site_urls.csv"
MODEL_PATH = "phishing_model.joblib"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# 2. Check the required columns
if not {"URL", "Label"}.issubset(df.columns):
    raise ValueError(
        f"Expected URL and Label columns. Found: {df.columns.tolist()}"
    )

# 3. Clean the data
df = df[["URL", "Label"]].dropna().copy()

df["URL"] = df["URL"].astype(str).str.strip()
df["Label"] = df["Label"].astype(str).str.strip().str.lower()

# Keep only known labels
df = df[df["Label"].isin(["good", "bad"])]
df = df[df["URL"] != ""]

# Remove exact duplicate URL records
df = df.drop_duplicates(subset=["URL"])

# bad = phishing, good = legitimate
df["Label"] = df["Label"].map({
    "good": 0,
    "bad": 1
})

print("Dataset size:", len(df))
print("Label counts:")
print(df["Label"].value_counts())

# 4. Separate URLs and labels
X = df["URL"]
y = df["Label"]

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining URLs:", len(X_train))
print("Testing URLs:", len(X_test))

# 6. Build the machine-learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=200000,
            sublinear_tf=True,
            dtype="float32"
        )
    ),
    (
        "classifier",
        SGDClassifier(
            loss="log_loss",
            max_iter=20,
            tol=1e-3,
            class_weight="balanced",
            random_state=42
        )
    )
])

# 7. Train the model
print("\nTraining model. Please wait...")
model.fit(X_train, y_train)

# 8. Evaluate on unseen test data
print("\nEvaluating model...")

predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
)

# 9. Save the trained model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")

# 10. Test the example URL
test_url = (
    "www.dghjdgf.com/paypal.co.uk/"
    "cycgi-bin/webscrcmd=_home-customer&nav=1/loading.php"
)

prediction = model.predict([test_url])[0]
probabilities = model.predict_proba([test_url])[0]
phishing_index = list(model.classes_).index(1)
phishing_score = probabilities[phishing_index] * 100

print("\nExample URL test:")
print("URL:", test_url)
print("Prediction:", "PHISHING" if prediction == 1 else "LEGITIMATE")
print(f"Model phishing score: {phishing_score:.2f}%")
