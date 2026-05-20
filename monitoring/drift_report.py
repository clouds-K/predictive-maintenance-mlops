"""
Module de détection de drift avec Evidently AI 0.7.x.
Auteur: Khouloud
"""

import pandas as pd
from pathlib import Path
from evidently.future.report import Report
from evidently.future.presets import DataDriftPreset


def generate_drift_report(
    reference_path: str = "data/processed/train.csv",
    current_path: str = "data/processed/test.csv",
    output_dir: str = "monitoring",
) -> None:
    """Génère un rapport de drift entre train et test.

    Args:
        reference_path: Données de référence (train).
        current_path: Données actuelles (test).
        output_dir: Dossier de sortie.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    target = "Machine failure"
    feature_cols = [c for c in reference.columns if c != target]

    print(f" Données : référence={reference.shape}, actuel={current.shape}")

    report = Report([DataDriftPreset()])

    my_eval = report.run(
        reference_data=reference[feature_cols],
        current_data=current[feature_cols],
    )

    my_eval.save_html(f"{output_dir}/drift_report.html")
    print(f" Rapport sauvegardé : {output_dir}/drift_report.html")


if __name__ == "__main__":
    generate_drift_report()