from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 11, 11),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG('airline_data_processing',
          default_args=default_args,
          description='Process airline data',
          schedule_interval=timedelta(days=1))

# Define tasks for each airline
airlines = ['DL', 'AA', 'UA', ...]  # list of airlines
tasks = {}

for airline in airlines:
    task_id = f'process_{airline}_data'
    bash_command = f'/home/zhuyiming/cour_P/workspace/ML_airfilghts/filter_flights.py {airline}'
    tasks[airline] = BashOperator(
        task_id=task_id,
        bash_command=bash_command,
        dag=dag
    )
