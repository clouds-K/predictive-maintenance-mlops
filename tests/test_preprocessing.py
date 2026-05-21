"""
Tests unitaires pour le module de prétraitement.
Auteur: Khouloud
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import clean_data, encode_features, split_data, scale_features


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture : DataFrame de test simulant AI4I 2020."""
    return pd.DataFrame(
        {
            "UDI": range(1, 101),
            "Product ID": [f"M{i}" for i in range(100)],
            "Type": ["L"] * 50 + ["M"] * 30 + ["H"] * 20,
            "Air temperature K": np.random.normal(300, 2, 100),
            "Process temperature K": np.random.normal(310, 1, 100),
            "Rotational speed rpm": np.random.normal(1500, 100, 100),
            "Torque Nm": np.random.normal(40, 5, 100),
            "Tool wear min": np.random.randint(0, 250, 100),
            "Machine failure": [0] * 97 + [1] * 3,
        }
    )


def test_clean_data_removes_useless_columns(sample_df: pd.DataFrame) -> None:
    """Vérifie que clean_data supprime les colonnes UDI et Product ID."""
    df_clean = clean_data(sample_df)
    assert "UDI" not in df_clean.columns
    assert "Product ID" not in df_clean.columns


def test_clean_data_no_duplicates(sample_df: pd.DataFrame) -> None:
    """Vérifie qu'il n'y a pas de doublons après nettoyage."""
    df_clean = clean_data(sample_df)
    assert df_clean.duplicated().sum() == 0


def test_encode_features_type_is_numeric(sample_df: pd.DataFrame) -> None:
    """Vérifie que la colonne Type est encodée en entier."""
    df_clean = clean_data(sample_df)
    df_encoded = encode_features(df_clean)
    assert pd.api.types.is_integer_dtype(df_encoded["Type"])


def test_encode_features_values_in_range(sample_df: pd.DataFrame) -> None:
    """Vérifie que les valeurs encodées sont dans {0, 1, 2}."""
    df_clean = clean_data(sample_df)
    df_encoded = encode_features(df_clean)
    assert set(df_encoded["Type"].unique()).issubset({0, 1, 2})


def test_split_data_sizes(sample_df: pd.DataFrame) -> None:
    """Vérifie que le split respecte le ratio 80/20."""
    df_clean = clean_data(sample_df)
    df_encoded = encode_features(df_clean)
    X_train, X_test, y_train, y_test = split_data(df_encoded)
    assert len(X_train) == 80
    assert len(X_test) == 20


def test_split_data_stratified(sample_df: pd.DataFrame) -> None:
    """Vérifie que le split est stratifié (taux de pannes similaire)."""
    df_clean = clean_data(sample_df)
    df_encoded = encode_features(df_clean)
    X_train, X_test, y_train, y_test = split_data(df_encoded)
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.05


def test_split_data_no_leakage(sample_df: pd.DataFrame) -> None:
    """Vérifie qu'il n'y a pas de data leakage entre train et test."""
    df_clean = clean_data(sample_df)
    df_encoded = encode_features(df_clean)
    X_train, X_test, _, _ = split_data(df_encoded)
    common = pd.merge(X_train, X_test, how="inner")
    assert len(common) == 0


def test_scale_features_mean_near_zero(sample_df: pd.DataFrame) -> None:
    """Vérifie que les features scalées ont une moyenne proche de 0."""
    df_clean = clean_data(sample_df)
    df_encoded = encode_features(df_clean)
    X_train, X_test, _, _ = split_data(df_encoded)
    X_train_sc, _ = scale_features(X_train, X_test)
    means = np.abs(X_train_sc.mean(axis=0))
    assert np.all(means < 0.1)


def test_no_missing_values_after_preprocessing(sample_df: pd.DataFrame) -> None:
    """Vérifie qu'il n'y a aucune valeur manquante après prétraitement."""
    df_clean = clean_data(sample_df)
    df_encoded = encode_features(df_clean)
    assert df_encoded.isnull().sum().sum() == 0
