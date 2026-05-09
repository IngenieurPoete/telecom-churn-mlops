"""
test_preprocess.py
Tests unitaires du pipeline de preprocessing.
Approche TDD adaptée au ML.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.preprocess import load_and_clean, encode_and_scale

# ── Fixture : données chargées une seule fois pour tous les tests ──
@pytest.fixture(scope="module")
def raw_df():
    return load_and_clean("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

@pytest.fixture(scope="module")
def split_data(raw_df):
    return encode_and_scale(raw_df.copy())

# ══════════════════════════════════════════════════════════════════
# TESTS SUR load_and_clean()
# ══════════════════════════════════════════════════════════════════

def test_totalcharges_is_float(raw_df):
    """TotalCharges doit être float64 après correction du bug."""
    assert raw_df['TotalCharges'].dtype == np.float64, \
        "TotalCharges doit être float64 — bug de typage non corrigé"

def test_no_missing_values(raw_df):
    """Aucune valeur manquante après nettoyage."""
    assert raw_df.isnull().sum().sum() == 0, \
        "Des valeurs manquantes subsistent après nettoyage"

def test_customerid_removed(raw_df):
    """customerID non prédictif — doit être supprimé."""
    assert 'customerID' not in raw_df.columns, \
        "customerID doit être supprimé du dataset"

def test_dataset_shape(raw_df):
    """Le dataset doit avoir 7043 lignes après nettoyage."""
    assert raw_df.shape[0] == 7043, \
        f"Nombre de lignes inattendu : {raw_df.shape[0]}"

# ══════════════════════════════════════════════════════════════════
# TESTS SUR encode_and_scale()
# ══════════════════════════════════════════════════════════════════

def test_stratified_split(split_data):
    """
    Le split doit être stratifié.
    Le taux de churn dans train ET test doit être ~26.5%.
    Bug classique ML : split non stratifié sur données déséquilibrées.
    """
    _, _, y_train, y_test, _ = split_data
    train_rate = y_train.mean()
    test_rate  = y_test.mean()

    assert abs(train_rate - 0.265) < 0.01, \
        f"Taux churn train incorrect : {train_rate:.3f} (attendu ~0.265)"
    assert abs(test_rate - 0.265) < 0.01, \
        f"Taux churn test incorrect : {test_rate:.3f} (attendu ~0.265)"

def test_churn_binary_encoded(split_data):
    """Churn doit être encodé en 0/1 — pas 'Yes'/'No'."""
    _, _, y_train, _, _ = split_data
    unique_values = set(y_train.unique())
    assert unique_values == {0, 1}, \
        f"Churn mal encodé : valeurs trouvées = {unique_values}"

def test_no_data_leakage(split_data):
    """
    Test anti Data Leakage.
    Le scaler doit être fitté UNIQUEMENT sur le train.
    Vérification : mean du train ≈ 0 après scaling (StandardScaler).
    """
    X_train, X_test, _, _, scaler = split_data
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    train_means = X_train[num_cols].mean()
    for col in num_cols:
        assert abs(train_means[col]) < 0.01, \
            f"Data Leakage possible sur {col} — mean train = {train_means[col]:.4f}"

def test_split_sizes(split_data):
    """80% train / 20% test."""
    X_train, X_test, _, _, _ = split_data
    total = len(X_train) + len(X_test)
    assert abs(len(X_train) / total - 0.8) < 0.01, \
        "Le ratio 80/20 n'est pas respecté"

def test_no_target_in_features(split_data):
    """Bug classique : la colonne Churn ne doit pas être dans les features."""
    X_train, _, _, _, _ = split_data
    assert 'Churn' not in X_train.columns, \
        "Data Leakage critique : Churn présent dans les features !"