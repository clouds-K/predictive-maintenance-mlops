"""
Module de prétraitement du dataset AI4I 2020.
Auteur: Khouloud
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset brut.

    Args:
        path: Chemin vers le fichier CSV.

    Returns:
        DataFrame pandas.
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le dataset : supprime colonnes inutiles et doublons.

    Args:
        df: DataFrame brut.

    Returns:
        DataFrame nettoyé.
    """
    # Supprimer colonnes non pertinentes pour le modèle
    df = df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])

    # Supprimer les doublons
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f" Doublons supprimés : {initial_len - len(df)}")
    # Renommer les colonnes pour compatibilité XGBoost
    df.columns = df.columns.str.replace(r"[\[\]<]", "", regex=True).str.strip()
    print(f" Colonnes renommées : {df.columns.tolist()}")


    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les variables catégorielles.

    Args:
        df: DataFrame nettoyé.

    Returns:
        DataFrame encodé.
    """
    le = LabelEncoder()
    df["Type"] = le.fit_transform(df["Type"])  # H=0, L=1, M=2
    print(f" Encodage Type : {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df


def split_data(
    df: pd.DataFrame,
    target: str = "Machine failure",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Sépare les données en train/test de façon stratifiée.

    Args:
        df: DataFrame complet.
        target: Nom de la colonne cible.
        test_size: Proportion du jeu de test.
        random_state: Graine aléatoire.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Split stratifié pour respecter le déséquilibre des classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f" Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")
    print(f" Taux pannes train : {y_train.mean()*100:.2f}% | test : {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str = "models",
) -> tuple:
    """Normalise les features numériques avec StandardScaler.

    Args:
        X_train: Features d'entraînement.
        X_test: Features de test.
        output_dir: Dossier pour sauvegarder le scaler.

    Returns:
        Tuple (X_train_scaled, X_test_scaled).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sauvegarder le scaler pour la phase d'inférence
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    print(f" Scaler sauvegardé dans {output_dir}/scaler.pkl")

    return X_train_scaled, X_test_scaled


def save_processed_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list,
    output_dir: str = "data/processed",
) -> None:
    """Sauvegarde les données prétraitées en CSV.

    Args:
        X_train: Features train scalées.
        X_test: Features test scalées.
        y_train: Labels train.
        y_test: Labels test.
        feature_names: Noms des colonnes.
        output_dir: Dossier de sortie.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train, columns=feature_names).assign(
        **{"Machine failure": y_train.values}
    ).to_csv(f"{output_dir}/train.csv", index=False)

    pd.DataFrame(X_test, columns=feature_names).assign(
        **{"Machine failure": y_test.values}
    ).to_csv(f"{output_dir}/test.csv", index=False)

    print(f" Données sauvegardées dans {output_dir}/")


if __name__ == "__main__":
    df = load_data("data/raw/ai4i2020.csv")
    df = clean_data(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_sc, X_test_sc = scale_features(X_train, X_test)
    save_processed_data(
        X_train_sc, X_test_sc, y_train, y_test,
        feature_names=X_train.columns.tolist()
    )
    print(" Prétraitement terminé !")