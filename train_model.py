
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, probabilities[:, list(model.classes_).index(1)])

print("=" * 55)
print("BREAST CANCER MODEL EVALUATION")
print("=" * 55)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"ROC-AUC (benign class): {auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Malignant (0)", "Benign (1)"],
        zero_division=0,
    )
)

print("Confusion Matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Save model in the project directory
model_path = Path(__file__).resolve().parent / "breast_cancer_model.pkl"
joblib.dump(model, model_path)

print(f"\nModel saved successfully: {model_path}")
