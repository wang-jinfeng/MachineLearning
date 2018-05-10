#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/4/28 下午5:49
    Author  : wangjf
    File    : svmMLiA.py
    GitHub  : https://github.com/wjf0627
"""
import random

from numpy import *


def loadDataSet(fileName):
    """
    Desc:
        对文件进行逐行解析，从而得到每行的类标签和整个特征矩阵
    :param fileName:
        -- 文件名
    :return:
        dataMat 特征矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """
    Desc:
        随机选择一个整数
    :param i:
        -- 第一个 alpha 的下标
    :param m:
        -- 返回一个不为 i 的随机数，在 0 ~ m 之间的整数值
    :return:
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    """
    Desc:
        clipAlpha（调整aj的值，使aj处于 L <= aj <= H）
    :param aj:
        -- 目标值
    :param H:
        -- 最大值
    :param L:
        -- 最小值
    :return:
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, totel, maxIter):
    """
    Desc:
        smoSimple
    :param dataMatIn:
        -- 数据集
    :param classLabels:
        -- 标签类别
    :param C:
        -- 松弛变量(常量值)，允许有些数据点可以处于分割面的错误一侧。控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
           可以通过调节该参数达到不同的效果。
    :param totel:
        -- 容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的效率）
    :param maxIter:
        -- 退出前最大的循环次数
    :return:
        b       模型的常量值
        alphas  拉格朗日乘子
    """
    # 矩阵转置 和 .T 一样的功能
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    # 初始化 b 和 alphas （alphas有点类似权重值）
    b = 0
    alphas = mat(zeros((m, 1)))
    # 没有任何 alpha 改变的情况下遍历数据的次数
    iter = 0
    while iter < maxIter:
        # 记录 alpha 是否已经进行优化，每次循环时设为 0，然后再对整个集合顺序遍历
        alphaPairsChanged = 0
        for i in range(m):
            # 我们预测的类别 y = w^Tx[i]+b;其中因为 w = ∑(1~n) a[n]*lable[n]*x[n]
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 预测结果和真实结果对比，计算误差 Ei
            Ei = fXi - float(labelMat[i])
            # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
            # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
            # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            if ((labelMat[i] * Ei < - totel) and (alphas[i] < C)) or ((labelMat[i] * Ei > totel) and (alphas[i] > 0)):
                # 如果满足优化的条件，我们就随机选取非 i 的一个点，进行优化比较
                j = selectJrand(i, m)
                # 预测 j 的结果
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # L 和 H 用于将 alphas[j] 调整到 0~C 之间。如果 L == H，就不做任何改变，直接进行 continue 语句
                # labelMat[i] != lableMat[j] 表示异侧，就相减，否则是同侧，就相加
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果相同，就没法优化了
                if L == H:
                    print("L == H")
                    continue
                # eta 是 alphas[i] 的最优修改量，如果 eta == 0，需要退出 for 循环的当前迭代过程
                # 参考《统计学习方法》李航-P125~P128<序列最小优化算法>
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T

                if eta >= 0:
                    print("eta>=0")
                    continue
                # 计算出一个新的 alphas[j] 的值
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 并使用辅助函数，以及 L 和 H 对其进行调整
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 检查 alphas[i] 是否只是轻微的改变，如果是的话，就退出 for 循环
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                # 然后 alphas[i] 和 alphas[j] 同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 在对 alphas[i] 和 alphas[j] 进行优化之后，给这两个 alphas 值设置一个常数 b
                # w = ∑[1~n] ai * yi * xi => b = yj -∑[1~n] ai * yi(xi * xj)
                # 所以：b1 - b = (y1 - y) - ∑[1~n] yi * (a1 - a) * (xi * x1)
                # 为什么要减2遍？因为是 减去∑[1~n]，正好2个变量 i 和 j，所以减2遍
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter:%d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        # 在 for 循环外，检查 alpha 值是否做了更新，如果在更新则将 iter 设为 0 后继续运行程序
        # 直到更新完毕后，iter 次循环无变化，才退出循环
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


def testSMO():
    dataArr, labelArr = loadDataSet("/Users/wangjf/Downloads/machinelearninginaction/Ch06/testSet.txt")
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 50)
    # labelArr
    print("b======", b)
    print(alphas[alphas > 0])


if __name__ == '__main__':
    testSMO()
