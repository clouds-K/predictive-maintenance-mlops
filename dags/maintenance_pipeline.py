"""
DAG Apache Airflow - Pipeline MLOps Predictive Maintenance.
Auteur: Khouloud
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess
import sys


default_args = {
    "owner": "khouloud",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def run_ingestion() -> None:
    """Tâche 1 : Ingestion et EDA des données."""
    subprocess.run(
        [sys.executable, "/opt/airflow/src/ingestion.py"],
        check=True
    )


def run_preprocessing() -> None:
    """Tâche 2 : Nettoyage, encodage, split et scaling."""
    subprocess.run(
        [sys.executable, "/opt/airflow/src/preprocessing.py"],
        check=True
    )


def run_training() -> None:
    """Tâche 3 : Entraînement des modèles avec MLflow tracking."""
    subprocess.run(
        [sys.executable, "/opt/airflow/src/train.py"],
        check=True
    )


def run_evaluation() -> None:
    """Tâche 4 : Évaluation finale du meilleur modèle."""
    subprocess.run(
        [sys.executable, "/opt/airflow/src/evaluate.py"],
        check=True
    )


with DAG(
    dag_id="predictive_maintenance_pipeline",
    default_args=default_args,
    description="Pipeline MLOps complet - Predictive Maintenance AI4I 2020",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "predictive-maintenance"],
) as dag:

    task_ingestion = PythonOperator(
        task_id="ingestion_eda",
        python_callable=run_ingestion,
    )

    task_preprocessing = PythonOperator(
        task_id="preprocessing",
        python_callable=run_preprocessing,
    )

    task_training = PythonOperator(
        task_id="model_training",
        python_callable=run_training,
    )

    task_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=run_evaluation,
    )

    # Définition de l'ordre d'exécution
    task_ingestion >> task_preprocessing >> task_training >> task_evaluation