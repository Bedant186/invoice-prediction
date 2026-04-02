import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/invoices.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "invoice_model.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Features and target
    feature_columns = [
        "invoice_amount",
        "payment_terms",
        "customer_score",
        "previous_delay_avg",
        "invoice_month",
        "days_to_due",
    ]
    target_column = "late_payment"

    X = df[feature_columns]
    y = df[target_column]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and feature columns
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURE_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Feature columns saved to: {FEATURE_PATH}")


if __name__ == "__main__":
    main()