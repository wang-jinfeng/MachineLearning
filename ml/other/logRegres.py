#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from numpy import *

"""
    Time    : 2018/4/27 下午5:44
    Author  : wangjf
    File    : logRegres.py
    GitHub  : https://github.com/wjf0627
"""


# 使用 Logistic 回归在简单数据集上的分类

# 解析数据
def loadDataSet(file_name):
    """
    Desc:
        加载并解析数据
    Args:
        file_name -- 文件名称，要解析的文件所在磁盘位置
    Returns:
        dataMat -- 原始数据的特征
        labelMat -- 原始数据的标签，也就是每条样本对应的类别
    """
    dataMat = []
    labelMat = []
    fr = open(file_name)
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 为了方便计算，我们将 X0 的值设为 1.0，也就是在每一行的开头添加 1.0 作为 X0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid 跳跃函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 正常的处理方案
# 两个参数：
#   第一个参数==> dataMatIn 是一个2维NumPy数组，每列分别代表每个不同的特征，每行则代表每个训练样本。
#   第二个参数==> classLabels 是类别标签，它是一个 1*100 的行向量。为了便于矩阵计算，需要将该行向量转换为列向量，做法是将原向量转置，
#       再将它赋值给labelMat。
def gradAscent(dataMatIn, classLabels):
    """
    Desc:
        正常的梯度上升法
    Args:
        dataMatIn -- 输入的 数据的特征 List
        classLabels -- 输入的数据的类别标签
    Returns:
        array(weights) -- 得到的最佳回归系数
    """
    # 转化为矩阵 [[1,1,2],[1,1,2]....]
    dataMatrix = mat(dataMatIn)  # 转化为 NumPy 矩阵
    # 转化为矩阵 [[0,1,0,1,0,1....]]，并转置 [[0],[1],[0],.....]
    # transpose() 行列转置函数
    # 将行向量转化为列向量 ==> 矩阵的转置
    labelMat = mat(classLabels).transpose()  # 首先将函数转换为 NumPy 矩阵，然后再将行向量转置为列向量
    # m -> 数据量，样本量 n -> 特征数
    m, n = shape(dataMatrix)
    print(m, n, '__' * 10, shape(dataMatrix.transpose()), '__' * 100)
    # alpha 代表向目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # 生成一个长度和特征数相同的矩阵，此处n为3 -> [[1],[1],[1]]
    # weights 代表回归系数， 此处的 ones((n,1)) 创建一个长度和特征数相同的矩阵，其中的数全部都是 1
    weights = ones((n, 1))
    for k in range(maxCycles):
        # m*3 的矩阵 * 3*1 的单位矩阵 ＝ m*1的矩阵
        # 那么乘上单位矩阵的意义，就代表：通过公式得到的理论值
        # 参考地址： 矩阵乘法的本质是什么？ https://www.zhihu.com/question/21351965/answer/31050145
        # print 'dataMatrix====', dataMatrix
        # print 'weights====', weights
        # n*3   *  3*1  = n*1
        h = sigmoid(dataMatrix * weights)
        # labelMat 是实际值
        error = (labelMat - h)  # 向量相减
        # 0.001* (3*m)*(m*1) 表示在每一个列上的一个误差情况，最后得出 x1,x2,xn的系数的偏移量
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# 可视化展示
def plotBestFit(dataArr, labelMat, weights):
    """
    Desc:
        将我们得到的数据可视化展示出来
    Args:
        dataArr:样本数据的特征
        labelMat:样本数据的类别标签，即目标变量
        weight:回归系数
    :return:
        None
    """
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    """
    y的由来，卧槽，是不是没看懂？
    首先理论上是这个样子的。
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# 随机梯度上升
# 梯度上升优化算法在每次更新数据集时都需要遍历整个数据集，计算复杂都较高
# 随机梯度上升一次只用一个样本点来更新回归系数
def stocGradAscent0(dataMatrix, classLabels):
    """
    Desc:
        随机梯度上升，只使用一个样本点来更新回归系数
    :param dataMatrix:
        -- 输入数据的数据特征（除去最后一列）
    :param classLabels:
        -- 输入数据的类别标签（最后一列数据）
    :return:
        weights -- 得到的最佳回归系数
    """
    m, n = shape(dataMatrix)
    alpha = 0.01
    # n * 1 的矩阵
    # 函数 ones 创建一个全为 1 的数组
    weights = ones(n)  # 初始化长度为 n 的数组，元素全为 1
    for i in range(m):
        # sum(dataMatrix[i] * weights) 为了求 f(x) 的值，f(x) = a1 * x1 + a2 * x2 + .. + an * xn，此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(dataMatrix[i] * weights))
        # 计算真实类别与预测类别之间的差值，然后按照该差值调整回归系数
        error = classLabels[i] - h
        # 0.01 * (1 * 1) * (1 * n)
        weights = weights + alpha * error * dataMatrix[i]
        print(weights, "*" * 10, dataMatrix[i], "*" * 10, error)
    return weights


# 随机梯度上升算法（优化版）
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    Desc:
        改进版的随机梯度上升，使用随机的一个样本来更新回归系数
    Args:
    :param dataMatrix:
        -- 输入数据的数据特征（出去最后一列数据）
    :param classLabels:
        -- 输入数据的类别标签（最后一列数据）
    :param numIter:
        -- 迭代次数
    :return:
        weights -- 得到的最佳回归系数
    """
    m, n = shape(dataMatrix)
    weights = ones(n)  # 创建与列数相同的矩阵的系数矩阵，所有的元素都是 1
    # 随机梯度，循环 150，观察是否收敛
    for j in range(numIter):
        # [0,1,2,...,m-1]
        dataIndex = range(m)
        for i in range(m):
            # i 和 j 的不断增大，导致alpha 的值不断减少，但是不为 0
            alpha = 4 / (10.0 + j + i) + 0.0001  # alpha 会随着迭代不断减小，但永远不会减少到 0，因为后面还有一个常数项 0.0001
            # 随机产生一个 0 ~ len() 之间的值
            randIndex = int(random.uniform(0, len(dataIndex)))
            # sum(dataMatrix[i] * weights) 是为了求 f(x) 的值，f(x) = a1 * x1 + a2 * x2 + ... + an * xn
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            # del (dataIndex[randIndex])
    return weights


# 分类函数，根据回归系数和特征向量来计算 Sigmoid 的值
def classifyVector(inX, weights):
    """
    Desc:
        最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于 0.5 函数返回 1，否则返回 0
    :param inX:
        --  特征向量，features
    :param weights:
        --  根据梯度上升/随机梯度上升 计算得到的回归系数
    :return:
        如果 prod 计算大于 0.5 函数返回 1，否则返回 0
    """
    pro = sigmoid(sum(inX * weights))
    if pro > 0.5:
        return 1.0
    else:
        return 0.0


# 打开测试集和训练集，并对数据进行格式化处理
def colicTest():
    """
    Desc:
        打开测试集和训练集，并对数据进行格式化处理
    :return:
        errorRate -- 分辨错误率
    """
    frTrain = open("/Users/wangjf/Downloads/machinelearninginaction/Ch05/horseColicTraining.txt")
    frTest = open("/Users/wangjf/Downloads/machinelearninginaction/Ch05/horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    # 解析训练数据集中的数据特征和 labels
    # trainingSet 中存储训练数据集的特征，trainingLabels 存储训练数据集中样本对应的分类标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 使用改进后的随机梯度上升算法求得在此数据集上的最佳回归系数 trainWeights
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def simpleTest():
    # 1.收集并准备数据
    dataMat, labelMat = loadDataSet("/Users/wangjf/Downloads/machinelearninginaction/Ch05/testSet.txt")
    # 2.训练模型，  f(x)=a1*x1+b2*x2+..+nn*xn中 (a1,b2, .., nn).T的矩阵值
    # 因为数组没有是复制n份， array的乘法就是乘法
    dataArr = array(dataMat)
    # weights = gradAscent(dataArr, labelMat)
    # print('*' * 30, weights)
    # plotBestFit(dataArr, labelMat, weights.getA())
    weights = stocGradAscent1(array(dataArr), labelMat, 500)
    plotBestFit(dataArr, labelMat, weights)


# 调用colicTest() 10 次并求得结果的平均值
def multiTest():
    numTests = 100
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    # simpleTest()
    multiTest()
