"""
Module d'entraînement des modèles avec tracking MLflow.
Auteur: Khouloud
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier
from pathlib import Path
import joblib


def load_processed_data(data_dir: str = "data/processed") -> tuple:
    """Charge les données prétraitées.

    Args:
        data_dir: Dossier contenant train.csv et test.csv.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")

    target = "Machine failure"
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    print(f" Données chargées : Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test: np.ndarray, y_test: pd.Series) -> dict:
    """Calcule les métriques d'évaluation.

    Args:
        model: Modèle entraîné.
        X_test: Features de test.
        y_test: Labels de test.

    Returns:
        Dictionnaire des métriques.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


def train_and_track(
    model,
    model_name: str,
    params: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Entraîne un modèle et track les résultats dans MLflow.

    Args:
        model: Instance du modèle sklearn.
        model_name: Nom du modèle pour MLflow.
        params: Hyperparamètres à logger.
        X_train: Features d'entraînement.
        X_test: Features de test.
        y_train: Labels d'entraînement.
        y_test: Labels de test.

    Returns:
        Dictionnaire des métriques.
    """
    with mlflow.start_run(run_name=model_name):
        # Logger les paramètres
        mlflow.log_params(params)

        # Entraînement
        model.fit(X_train, y_train)

        # Évaluation
        metrics = evaluate_model(model, X_test, y_test)

        # Logger les métriques
        mlflow.log_metrics(metrics)

        # Sauvegarder le modèle dans MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f" {model_name}")
        print(f"   Accuracy  : {metrics['accuracy']:.4f}")
        print(f"   Precision : {metrics['precision']:.4f}")
        print(f"   Recall    : {metrics['recall']:.4f}")
        print(f"   F1-Score  : {metrics['f1']:.4f}")
        print(f"   ROC-AUC   : {metrics['roc_auc']:.4f}")

    return metrics


def save_best_model(
    models_metrics: dict,
    models: dict,
    output_dir: str = "models",
) -> None:
    """Sauvegarde le meilleur modèle selon le F1-Score.

    Args:
        models_metrics: Dictionnaire {nom: métriques}.
        models: Dictionnaire {nom: modèle entraîné}.
        output_dir: Dossier de sauvegarde.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    best_name = max(models_metrics, key=lambda k: models_metrics[k]["f1"])
    best_model = models[best_name]

    joblib.dump(best_model, f"{output_dir}/best_model.pkl")
    print(f" Meilleur modèle : {best_name}")
    print(f"   F1-Score : {models_metrics[best_name]['f1']:.4f}")
    print(f"   Recall   : {models_metrics[best_name]['recall']:.4f}")
    print(f" Modèle sauvegardé dans {output_dir}/best_model.pkl")


if __name__ == "__main__":
    mlflow.set_experiment("predictive-maintenance")

    X_train, X_test, y_train, y_test = load_processed_data()

    # Définir les modèles et leurs hyperparamètres
    models_config = {
        "LogisticRegression": {
            "model": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ),
            "params": {"class_weight": "balanced", "max_iter": 1000},
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "params": {"n_estimators": 100, "class_weight": "balanced"},
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=100,
                scale_pos_weight=9661 / 339,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            ),
            "params": {"n_estimators": 100, "scale_pos_weight": round(9661 / 339, 2)},
        },
    }

    all_metrics = {}
    all_models = {}

    print("🚀 Démarrage de l'entraînement...\n")
    for name, config in models_config.items():
        metrics = train_and_track(
            model=config["model"],
            model_name=name,
            params=config["params"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        all_metrics[name] = metrics
        all_models[name] = config["model"]

    save_best_model(all_metrics, all_models)
    print(" Entraînement terminé ! Lance : mlflow ui")
