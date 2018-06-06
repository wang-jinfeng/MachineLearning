#!/usr/bin/env python3
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
from airflow import HivePartitionSensor

from airflow import DAG

# -*- coding: utf-8 -*-
"""
    Time    : 2018/5/14 下午3:35
    Author  : wangjf
    File    : event_mid.py
    GitHub  : https://github.com/wjf0627
"""
# noinspection PyCallByClass
default_args = {
    'owner': 'wangjf',
    'depends_on_past': False,
    'start_date': datetime.strptime('2018-04-13', '%Y-%m-%d'),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}
dag = DAG(
    dag_id='event_mid',
    schedule_interval=timedelta(days=1),
    default_args=default_args,
    dagrun_timeout=timedelta(minutes=10))

tkio_sensor = HivePartitionSensor(
    task_id="check_tkio_partition",
    table="tkio_event",
    partition="ds= {{ execution_date }} ",
    metastore_conn_id='aws_hive_server',
    schema='dmp',
    dag=dag
)

tk_sensor = HivePartitionSensor(
    task_id="check_tk_partition",
    table="tk_event",
    partition="ds= {{ execution_date }} ",
    metastore_conn_id='aws_hive_server',
    schema='dmp',
    dag=dag
)

game_sensor = HivePartitionSensor(
    task_id="check_game_partition",
    table="game_event",
    partition="ds= {{ execution_date }} ",
    metastore_conn_id='aws_hive_server',
    schema='dmp',
    dag=dag
)

success = BashOperator(task_id='airflow_ssh',
                       depends_on_past=False,
                       bash_command="echo 'SUCCESS!!!'",
                       dag=dag)

tkio_sensor >> success
tk_sensor >> success
game_sensor >> success
