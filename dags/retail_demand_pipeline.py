# dags/retail_demand_pipeline.py

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "priyanshu",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2)
}

with DAG(
    dag_id="retail_demand_pipeline",
    default_args=default_args,
    description="Store Item Demand Forecasting MLOps Pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["forecasting", "retail", "mlops"]
) as dag:

    validate = BashOperator(
        task_id="validate_data",
        bash_command="python /opt/airflow/scripts/validate.py"
    )

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="python /opt/airflow/scripts/preprocess.py"
    )


    validate >> preprocess 