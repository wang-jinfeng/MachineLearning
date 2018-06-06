#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/5/11 上午11:49
    Author  : wangjf
    File    : svm-complete.py
    GitHub  : https://github.com/wjf0627
"""

class optStruct:
    """
    建立的数据结构来保存所有的重要值
    """
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        """
        :param dataMatIn:
            数据量
        :param classLabels:
            类别标签
        :param C:
            松弛变量（常量值），允许有些数据点可以处于分割面的错误一侧。控制最大化间隔和保证大部分的函数间隔小于 1.0 这两个目标的权重。
            可以通过调节该参数达到不同的效果。
        :param toler:
            容错率
        :param kTup:
            包含核函数信息的元组
        """
