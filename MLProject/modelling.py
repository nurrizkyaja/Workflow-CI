import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature
from pathlib import Path

def main():
    # Setup path
    base_dir = Path(__file__).parent
    csv_path = base_dir / "titanic_preprocessing" / "titanic_preprocessed_train.csv"

    # Load dataset
    df = pd.read_csv(csv_path)

    # Convert object columns to numeric labels
    label_cols = df.select_dtypes(include='object').columns
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # MLflow run
    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(5)

        # Custom conda_env to bypass Anaconda TOS error
        custom_conda_env = {
            "name": "mlflow-env",
            "channels": ["conda-forge", "nodefaults"],
            "dependencies": [
                "python=3.12.7",
                "pip",
                {
                    "pip": [
                        "mlflow==2.19.0",
                        "scikit-learn==1.6.1",
                        "pandas==2.2.3",
                        "fastapi",
                        "uvicorn"
                    ]
                }
            ]
        }

        # Log model to local artifact path
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            conda_env=custom_conda_env
        )

        print(f"✅ Training Selesai. Akurasi: {accuracy:.4f}")

if __name__ == "__main__":
    main()
