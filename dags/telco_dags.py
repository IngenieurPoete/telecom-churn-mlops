from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
# Importe tes fonctions (adapte les chemins si besoin)
# from src.data.validate_data import validate_dataset
# from src.data.preprocess import load_and_clean, encode_and_scale
# from src.models.train import train_model

default_args = {'owner': 'mlops', 'retries': 1}

with DAG('telco_mlops_pipeline', default_args=default_args, schedule_interval='@daily', start_date=datetime(2024, 1, 1), catchup=False) as dag:
    
    # Tâches mockées pour la démo
    validate = PythonOperator(task_id='validate', python_callable=lambda: print("Validation GE OK"))
    preprocess = PythonOperator(task_id='preprocess', python_callable=lambda: print("Preprocessing OK"))
    train = PythonOperator(task_id='train', python_callable=lambda: print("Training OK"))
    deploy = PythonOperator(task_id='deploy', python_callable=lambda: print("Deploy OK"))

    validate >> preprocess >> train >> deploy