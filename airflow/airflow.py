#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/5/3 下午4:40
    Author  : wangjf
    File    : airflow.py
    GitHub  : https://github.com/wjf0627
"""
import airflow
from datetime import timedelta

from airflow import DAG
from airflow.operators.bash_operator import BashOperator

default_args = {
    'owner': 'wangjf',
    'depends_on_past': False,
    'start_date': airflow.utils.dates.days_ago(2),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'airflow',
    schedule_interval=timedelta(days=1),
    default_args=default_args
)

command = "ssh -i /var/lib/spark/wangjf/data/bpu-test-new.pem ec2-user@10.1.28.107 pwd "

BashOperator(task_id='airflow_ssh',
             depends_on_past=False,
             bash_command=command,
             dag=dag)
