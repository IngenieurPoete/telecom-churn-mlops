"""
train_mlflow.py
Entraînement de 3 modèles avec tracking MLflow complet.
"""
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, roc_auc_score,
                             precision_score, recall_score)
import joblib
import os

# ── Chargement des données preprocessées ──────────────────────────
X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

# ── Définition des modèles à comparer ─────────────────────────────
MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}

# ── Experiment MLflow ──────────────────────────────────────────────
mlflow.set_experiment("telco-churn-prediction")

best_f1    = 0
best_model = None
best_name  = ""

for model_name, model in MODELS.items():

    with mlflow.start_run(run_name=model_name):

        # Entraînement
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calcul des métriques
        metrics = {
            "f1_score":  round(f1_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        }

        # ── Log dans MLflow ──
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"  F1={metrics['f1_score']} | Recall={metrics['recall']} | AUC={metrics['auc_roc']}")

        # Sélection du meilleur modèle (critère : Recall)
        if metrics["recall"] > best_f1:
            best_f1    = metrics["recall"]
            best_model = model
            best_name  = model_name

# ── Sauvegarde du meilleur modèle ─────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
print(f"\n✅ Meilleur modèle : {best_name} (Recall={best_f1:.4f})")
print("✅ Sauvegardé → models/best_model.pkl")