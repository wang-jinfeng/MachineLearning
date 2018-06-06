#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/5/11 下午6:22
    Author  : wangjf
    File    : hive_partition_sensor.py
    GitHub  : https://github.com/wjf0627
"""

from airflow.operators.sensors import BaseSensorOperator
from airflow.utils.decorators import apply_defaults


class HivePartitionSensor(BaseSensorOperator):
    """
    Waits for a partition to show up in Hive.
    Note: Because ``partition`` supports general logical operators, it
    can be inefficient. Consider using NamedHivePartitionSensor instead if
    you don't need the full flexibility of HivePartitionSensor.
    :param table: The name of the table to wait for, supports the dot
        notation (my_database.my_table)
    :type table: string
    :param partition: The partition clause to wait for. This is passed as
        is to the metastore Thrift client ``get_partitions_by_filter`` method,
        and apparently supports SQL like notation as in ``ds='2015-01-01'
        AND type='value'`` and comparison operators as in ``"ds>=2015-01-01"``
    :type partition: string
    :param metastore_conn_id: reference to the metastore thrift service
        connection id
    :type metastore_conn_id: str
    """
    template_fields = ('schema', 'table', 'partition',)
    ui_color = '#C5CAE9'

    @apply_defaults
    def __init__(self,
                 table, partition="ds='{{ ds }}'",
                 metastore_conn_id='metastore_default',
                 schema='default',
                 poke_interval=60 * 3,
                 *args,
                 **kwargs):
        super(HivePartitionSensor, self).__init__(
            poke_interval=poke_interval, *args, **kwargs)
        if not partition:
            partition = "ds='{{ ds }}'"
        self.metastore_conn_id = metastore_conn_id
        self.table = table
        self.partition = partition
        self.schema = schema

    def poke(self, context):
        if '.' in self.table:
            self.schema, self.table = self.table.split('.')
        self.log.info(
            'Poking for table {self.schema}.{self.table}, '
            'partition {self.partition}'.format(**locals()))
        if not hasattr(self, 'hook'):
            from airflow.hooks.hive_hooks import HiveMetastoreHook
            self.hook = HiveMetastoreHook(
                metastore_conn_id=self.metastore_conn_id)
        return self.hook.check_for_partition(
            self.schema, self.table, self.partition)
