import joblib
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_models():
    models = {}
    model_dir = "models"
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl") and filename != "scaler.pkl":
            model_name = filename.replace(".pkl", "").replace("_", " ").title()
            model = joblib.load(os.path.join(model_dir, filename))
            models[model_name] = model

    return {"models": models, "scaler": scaler}

def preprocess_input(df):
    scaler = joblib.load("models/scaler.pkl")
    return scaler.transform(df)

def evaluate_models(loaded, X):
    models = loaded["models"]
    results = {}
    kpis = {}

    for name, model in models.items():
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

        kpis[name] = {
            "Accuracy": accuracy_score([1], y_pred),
            "Precision": precision_score([1], y_pred),
            "Recall": recall_score([1], y_pred),
            "F1 Score": f1_score([1], y_pred),
            "AUC": roc_auc_score([1], y_prob) if len(set(y_pred)) > 1 else "N/A"
        }

        results[name] = int(y_pred[0])

    return results, kpis
