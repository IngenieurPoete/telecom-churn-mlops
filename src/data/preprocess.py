"""
preprocess.py
Nettoyage, encodage et split du dataset Telco Churn.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Charge le CSV et corrige le bug TotalCharges.

    Args:
        filepath: Chemin vers le CSV brut.

    Returns:
        DataFrame nettoyé.
    """
    df = pd.read_csv(filepath)

    # Bug connu : TotalCharges est object à cause des espaces
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Clients tenure=0 → TotalCharges impute à 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # customerID non prédictif
    df.drop(columns=['customerID'], inplace=True)

    return df


def encode_and_scale(df: pd.DataFrame):
    """
    Encode les variables catégorielles et scale les numériques.

    Args:
        df: DataFrame nettoyé.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Cible : Yes → 1, No → 0
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # Variables binaires Yes/No
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    # One-Hot Encoding des catégorielles restantes
    cat_cols = ['gender', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Features et cible
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Split STRATIFIÉ (crucial car déséquilibre 73/27)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y          # ← garantit 27% de churners dans les deux sets
    )

    # Scaling des variables numériques
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_and_clean("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test, scaler = encode_and_scale(df)

    # Sauvegarde pour réutilisation
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"✅ Train : {X_train.shape} | Test : {X_test.shape}")
    print(f"✅ Taux churn train : {y_train.mean()*100:.1f}%")
    print(f"✅ Taux churn test  : {y_test.mean()*100:.1f}%")
    print("✅ Scaler sauvegardé → models/scaler.pkl")