

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import joblib
import json
import os

def train_model():
    """Charge les données preprocessées, entraîne et sauvegarde le modèle."""

    # Chargement
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Entraînement
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'   # compense le déséquilibre 73/27
    )
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    print(f"F1-Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    # Sauvegarde modèle + métriques
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/churn_model.pkl")

    metrics = {"f1_score": round(f1, 4), "auc_roc": round(auc, 4)}
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("✅ Modèle sauvegardé → models/churn_model.pkl")
    return model, metrics

if __name__ == "__main__":
    train_model()