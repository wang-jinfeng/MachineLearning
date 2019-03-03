#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/7/4 下午5:20
    Author  : wangjf
    File    : kMeansSklearn.py
    GitHub  : https://github.com/wjf0627
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

dataMat = []
fr = open('/Users/wangjf/WorkSpace/AiLearning/data/10.KMeans/testSet.txt')
for line in fr.readlines():
    curLine = line.strip().split('\t')
    fltLine = list(map(float, curLine))
    dataMat.append(fltLine)

#   训练模型
km = KMeans(n_clusters=4)
km.fit(dataMat)
km_pred = km.predict(dataMat)  # 预测
centers = km.cluster_centers_  # 质心

#   可视化结果
plt.scatter(np.array(dataMat)[:, 1], np.array(dataMat)[:, 0], c=km_pred)
plt.scatter(centers[:, 1], centers[:, 0], c='r')

plt.show()
