# -*- coding:utf-8 -*-
from numpy import *
from os import listdir
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


"""
对于每一个在数据集中的数据点：
    计算目标的数据点（需要分类的数据点）与该数据点的距离
    将距离排序：从小到大
    选取前K个最短距离
    选取这K个中最多的分类类别
    返回该类别来作为目标数据点的预测值
"""


# k-近邻算法
def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #   距离度量，度量公式为欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    #   将距离排序：从小到大
    sortedDistIndicies = distances.argsort()
    #   选取前K个最短距离，选取这K个中最多的分类类别
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


# 将文本记录转化为NumPy的解析程序
def file2matrix(filename):
    """
    param:
        filename:数据文件路径
    return:
        数据矩阵 returnMat 和对应的类别 classLabelVector
    from ml import kNN
    datingDataMat,datingLabels = kNN.file2matrix("/Users/wangjf/Downloads/machinelearninginaction/Ch02/datingTestSet2.txt")

    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    #   获得文件中的数据行的行数
    numberOfLines = len(arrayOLines)
    #   生成对应的空矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        #   返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
        #   以'\t'切割字符串
        listFromLine = line.split('\t')
        #   每列的属性数据
        returnMat[index, :] = listFromLine[0:3]
        #   每列的类别数据，就是label标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    #   返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    """
    :param dataSet:数据集
    :return:
        归一化后的数据集normDataSet,ranges和minVals即最小值与范围，并没有用到
    归一化公式:
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转换为 0 到 1 的区间
    """
    #   计算每种属性的最大值，最小值，范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #   极差
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    #   生成与最小值之差组成的矩阵
    normDataSet = normDataSet - tile(minVals, (m, 1))
    #   将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 分类器针对约会网站的测试代码
def datingClassTest():
    """
    对约会网站的测试方法
    :return:
        错误数
    """
    #   设置测试数据的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围，一部分测试一部分作为样本
    #   从文本中加载数据
    datingDataMat, datingLabels = file2matrix('/Users/wangjf/Downloads/machinelearninginaction/Ch02/datingTestSet2.txt')
    #   归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #   m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    #   设置测试的样本数量，numTestVecs:m 表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        #   对数据测试
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d,the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]: errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


# 约会网站测试代码
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('/Users/wangjf/Downloads/machinelearninginaction/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


# 手写识别系统
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readlines()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 手写数字识别系统的测试代码
def handwritingClassTest():
    #   导入训练数据
    hwLabels = []
    trainingFileList = listdir('/Users/wangjf/Downloads/machinelearninginaction/Ch02/digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    #   hwLabels存储0~9对应的index位置，trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        #   将 32*32 的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector(
            "/Users/wangjf/Downloads/machinelearninginaction/Ch02/digits/trainingDigits/%s" % fileNameStr)

    #   导入测试数据
    testFileList = listdir("/Users/wangjf/Downloads/machinelearninginaction/Ch02/digits/testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(
            '/Users/wangjf/Downloads/machinelearninginaction/Ch02/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d,the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr: errorCount += 1.0

    print('\nthe total number of errors is: %d' % errorCount)
    print('\nthe total error rate is: %f' % (errorCount / float(mTest)))


"""
k 值的选择
    1、k 值的选择会对 k 近邻算法的结果产生重大的影响。
    2、如果选择较小的 k 值，就相当于用较小的邻域中的训练实例进行预测，“学习”的近似误差（approximation error）会减小，只有与输入实例较近的
    （相似的）训练实例才会对预测结果起作用。但缺点是“学习”的估计误差（estimation error）会增大，预测结果会对近邻的实例点非常敏感。如果邻近
    的实例点恰巧是噪声，预测就会出错。换句话说，k 值的减小就意味着整体模型变得复杂，容易发生过拟合。
    3、如果选择较大的 k 值，就相当于用较大的邻域中的训练实例进行预测。其优点是可以减少学习的估计误差。但缺点是学习的近似误差会增大。这时与输
    入实例较远的（不相似的）训练实例也会对预测起作用，使预测发生错误。 k 值的增大就意味着整体的模型变得简单。
    4、近似误差和估计误差，请看这里：https://www.zhihu.com/question/60793482
    
距离度量
    1、特征空间中两个实例点的距离是两个实例点相似程度的反映。
    2、k 近邻模型的特征空间一般是 n 维实数向量空间 向量空间 。使用的距离是欧氏距离，但也可以是其他距离，如更一般的 Lp距离 距离，或者 Minkowski 距离。

分类决策规则
    1、k 近邻算法中的分类决策规则往往是多数表决，即由输入实例的 k 个邻近的训练实例中的多数类决定输入实例的类。
"""
