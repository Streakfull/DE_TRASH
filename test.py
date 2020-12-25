import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn
# from IPython.display import Javascript
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import minmax_scale
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from datetime import date
from utils.life_expectancy_df import transform_life_expectancy_df
from utils.happiness_df import transform_happiness

pd.set_option('display.max_columns', None)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2020, 12, 13)
}

dag = DAG(
    'ETL-Pie',
    default_args=default_args,
    description='ETL pipeline',
    schedule_interval='@once',
)


def load_data_sets(**kwargs):
    life_expectancy_df = pd.read_csv(
        "https://raw.githubusercontent.com/Alsouidan/shared-data-CSEN-1095/main/Life%20Expectancy%20Data.csv")
    countries_df = pd.read_csv(
        "https://raw.githubusercontent.com/Alsouidan/shared-data-CSEN-1095/main/250%20Country%20Data.csv")
    happiness_dfs = {"2015": 0, "2016": 1, "2017": 2, "2018": 3, "2019": 4}
    for key in happiness_dfs.keys():
        happiness_dfs[key] = pd.read_csv(
            "https://raw.githubusercontent.com/Alsouidan/shared-data-CSEN-1095/main/{year}.csv".format(year=key)).to_json()
    return life_expectancy_df.to_json(), countries_df.to_json(), happiness_dfs


def normalize_column_names(**context):
    life_expectancy_df, countries_df, happiness_dfs = context['task_instance'].xcom_pull(
        task_ids='load_data_sets')
    life_expectancy_df = pd.read_json(life_expectancy_df)
    countries_df = pd.read_json(countries_df)

    def rename_columns(df):
        df = df.rename(str.strip, axis='columns')
        df = df.rename(columns=lambda name:  " ".join(
            w[:1].upper() + w[1:] for w in name.split()))
        return df

    life_expectancy_df = rename_columns(life_expectancy_df).to_json()
    countries_df = rename_columns(countries_df).to_json()

    for key in happiness_dfs.keys():
        happiness_dfs[key] = rename_columns(
            pd.read_json(happiness_dfs[key])).to_json()

    return life_expectancy_df, countries_df, happiness_dfs


def transform_life_expectancy_callable(**context):
    life_expectancy_df, countries_df, happiness_dfs = context['task_instance'].xcom_pull(
        task_ids='normalize_column_names')
    life_expectancy_df = transform_life_expectancy_df(
        pd.read_json(life_expectancy_df))
    return life_expectancy_df.to_json()


def transform_happiness_callable(**context):
    life_expectancy_df, countries_df, happiness_dfs = context['task_instance'].xcom_pull(
        task_ids='normalize_column_names')
    for key in happiness_dfs.keys():
        happiness_dfs[key] = pd.read_json(happiness_dfs[key])
    happiness_dfs = transform_happiness(happiness_dfs)
    print(happiness_dfs, "OKKK")
    for key in happiness_dfs.keys():
        happiness_dfs[key] = happiness_dfs[key].to_json()
    return happiness_dfs


load_data = PythonOperator(
    task_id='load_data_sets',
    provide_context=True,
    python_callable=load_data_sets,
    dag=dag,


)

normalize_task = PythonOperator(
    task_id='normalize_column_names',
    provide_context=True,
    python_callable=normalize_column_names,
    dag=dag,
)

transform_life_expectancy_task = PythonOperator(
    task_id="tansform_life_expectancy",
    provide_context=True,
    python_callable=transform_life_expectancy_callable,
    dag=dag
)

transform_happiness_task = PythonOperator(
    task_id="transform_happiness",
    provide_context=True,
    python_callable=transform_happiness_callable,
    dag=dag,
)


load_data >> normalize_task

normalize_task >> transform_life_expectancy_task

normalize_task >> transform_happiness_task
