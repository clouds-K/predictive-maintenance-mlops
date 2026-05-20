"""
Module d'évaluation finale du meilleur modèle.
Auteur: Khouloud
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate(
    model_path: str = "models/best_model.pkl",
    data_dir: str = "data/processed",
    output_dir: str = "data/processed",
) -> None:
    """Évalue le meilleur modèle et génère les visualisations.

    Args:
        model_path: Chemin vers le modèle sauvegardé.
        data_dir: Dossier des données prétraitées.
        output_dir: Dossier de sortie pour les figures.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Charger modèle et données
    model = joblib.load(model_path)
    test = pd.read_csv(f"{data_dir}/test.csv")
    X_test = test.drop(columns=["Machine failure"])
    y_test = test["Machine failure"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Rapport de classification
    print("===== RAPPORT DE CLASSIFICATION =====")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Panne"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Panne"],
                yticklabels=["Normal", "Panne"])
    plt.title("Matrice de Confusion - XGBoost")
    plt.ylabel("Réel")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    print("✅ Matrice de confusion sauvegardée")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        features = X_test.columns
        plt.figure(figsize=(8, 5))
        pd.Series(importances, index=features).sort_values().plot(
            kind="barh", color="steelblue"
        )
        plt.title("Importance des Features - XGBoost")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png")
        plt.close()
        print("✅ Feature importance sauvegardée")


if __name__ == "__main__":
    evaluate()