#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/5/11 下午5:08
    Author  : wangjf
    File    : HivePartitionSensor.py
    GitHub  : https://github.com/wjf0627
"""
import logging

from airflow.operators.sensors import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
from impala.dbapi import *


class HivePartitionSensor(BaseSensorOperator):
    """
    Waits for a partition to show up in Hive.

    :param host, port: the host and port of hiveserver2
    :param table: The name of the table to wait for, supports the dot notation (my_database.my_table)
    :type table: string
    :param partition: The partition clause to wait for. This is passed as
        is to the metastore Thrift client,and apparently supports SQL like
        notation as in ``ds='2016-12-01'``.
    :type partition: string
    """
    template_fields = ('table', 'partition',)
    ui_color = '#2b2d42'

    @apply_defaults
    def __init__(
            self,
            conn_host, conn_port,
            table, partition="ds='{{ ds }}'",
            poke_interval=60 * 3,
            *args, **kwargs):
        super(HivePartitionSensor, self).__init__(
            poke_interval=poke_interval, *args, **kwargs)
        if not partition:
            partition = "ds='{{ ds }}'"
        self.table = table
        self.partition = partition
        self.conn_host = conn_host
        self.conn_port = conn_port
        self.conn = connect(host=self.conn_host, port=self.conn_port, auth_mechanism='PLAIN')

    def poke(self, context):
        logging.info(
            'Poking for table {self.table}, '
            'partition {self.partition}'.format(**locals()))
        cursor = self.conn.cursor()
        cursor.execute("show partitions {}".format(self.table))
        partitions = cursor.fetchall()
        partitions = [i[0] for i in partitions]
        if self.partition in partitions:
            return True
        else:
            return False
