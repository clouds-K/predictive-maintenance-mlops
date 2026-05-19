"""
Module d'ingestion et d'EDA du dataset AI4I 2020.
Auteur: Khouloud
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset AI4I 2020.

    Args:
        path: Chemin vers le fichier CSV.

    Returns:
        DataFrame pandas chargé.
    """
    df = pd.read_csv(path)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def basic_info(df: pd.DataFrame) -> None:
    """Affiche les informations de base du dataset.

    Args:
        df: DataFrame à analyser.
    """
    print("\n===== INFOS GÉNÉRALES =====")
    print(df.info())
    print("\n===== STATISTIQUES =====")
    print(df.describe())
    print("\n===== VALEURS MANQUANTES =====")
    print(df.isnull().sum())
    print("\n===== DISTRIBUTION DE LA CIBLE =====")
    print(df["Machine failure"].value_counts())
    print(f"Taux de panne : {df['Machine failure'].mean()*100:.2f}%")


def plot_eda(df: pd.DataFrame, output_dir: str = "data/processed") -> None:
    """Génère les visualisations EDA.

    Args:
        df: DataFrame à visualiser.
        output_dir: Dossier de sortie pour les figures.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Distribution de la variable cible
    plt.figure(figsize=(6, 4))
    df["Machine failure"].value_counts().plot(kind="bar", color=["steelblue", "tomato"])
    plt.title("Distribution des pannes (0=Normal, 1=Panne)")
    plt.xlabel("Machine failure")
    plt.ylabel("Nombre d'observations")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/target_distribution.png")
    plt.close()
    print("✅ Figure 1 sauvegardée : target_distribution.png")

    # 2. Corrélation entre features numériques
    numeric_cols = ["Air temperature [K]", "Process temperature [K]",
                    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols + ["Machine failure"]].corr(),
                annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    print("✅ Figure 2 sauvegardée : correlation_matrix.png")

    # 3. Distribution des features numériques
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, col in enumerate(numeric_cols):
        ax = axes[i // 3][i % 3]
        df[col].hist(ax=ax, bins=30, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("Valeur")
        ax.set_ylabel("Fréquence")
    axes[1][2].axis("off")
    plt.suptitle("Distribution des features numériques", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/features_distribution.png")
    plt.close()
    print("✅ Figure 3 sauvegardée : features_distribution.png")

    # 4. Types de pannes
    failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    failure_counts = df[failure_types].sum()
    plt.figure(figsize=(7, 4))
    failure_counts.plot(kind="bar", color="tomato", edgecolor="white")
    plt.title("Répartition par type de panne")
    plt.xlabel("Type de panne")
    plt.ylabel("Nombre d'occurrences")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/failure_types.png")
    plt.close()
    print("✅ Figure 4 sauvegardée : failure_types.png")


if __name__ == "__main__":
    df = load_data("data/raw/ai4i2020.csv")
    basic_info(df)
    plot_eda(df)