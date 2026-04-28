# dags/retail_demand_pipeline.py

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default settings for all tasks
default_args = {
    "owner": "priyanshu",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2)
}

# Define the DAG
with DAG(
    dag_id="retail_demand_pipeline",
    is_paused_upon_creation=False,
    default_args=default_args,
    description="Store Item Demand Forecasting MLOps Pipeline",
    start_date=datetime(2026, 2, 2),
    schedule="@daily",
    catchup=False,
    tags=["forecasting", "retail", "mlops"]
) as dag:

    # Step 1: Validate raw input data
    validate = BashOperator(
        task_id="validate_data",
        bash_command="python /opt/airflow/scripts/validate.py"
    )

    # Step 2: Preprocess validated data
    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="python /opt/airflow/scripts/preprocess.py"
    )

    # Step 3: Generate baseline model metrics
    baseline = BashOperator(
        task_id="generate_baseline",
        bash_command="python /opt/airflow/scripts/generate_baseline.py"
    )

    # Task execution order
    validate >> preprocess >> baseline