#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/5/11 上午11:49
    Author  : wangjf
    File    : svm-complete.py
    GitHub  : https://github.com/wjf0627
"""
from numpy import *
import matplotlib.pyplot as plt


class optStruct:
    """
    建立的数据结构来保存所有的重要值
    """

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
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
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler

        #   数据的行数
        self.m = shape(dataMatIn)[0]
        self.alphs = mat(zeros((self.m, 1)))
        self.b = 0

        #   误差缓存，第一列给出的是 eCache 是否有效的标志位，第二列给出的是实际的 E 值
        self.eCache = mat(zeros((self.m, 2)))

        #   m行m列的矩阵
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i], kTup)


def kernelTrans(X, A, kTup):  # calc the kernel of transform data to a higher dimensional space
    """
    核转换函数
    :param X:
        dataMatIn 数据集
    :param A:
        dataMatIn 数据集的第 i 行的数据
    :param kTup:
        核函数的信息
    :return:
    """
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        #   linear kernel:  m*n * n*1 = m*1
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
            #   径向基函数的高斯版本
            K = exp(K / (-1 * kTup[1] ** 2))  # divide in Numpy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


def loadDataSet(fileName):
    """
    Desc:
        对文件进行逐行解析，从而得到每行的类标签和整个特征矩阵
    :param fileName:文件名
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


def calcEk(oS, k):
    """
    calcEk (求 Ek 误差，预测值-真实值的差)
    该过程在完整版的 SMO 算法中出现次数较多，因此将其单独作为一个方法
    :param k: 具体的某一行
    :param oS: optStruct对象
    :return:
        Ek 预测结果与真实结果比对，计算误差 Ek
    """
    fXK = multiply(oS.alphs, oS.labelMat).T * oS.K[:, k] + oS.b
    Ek = fXK - float(oS.labelMat[k])
    return Ek


def selectJrand(i, m):
    """
    随机选择一个整数
    :param i: 第一个 alpha 的下标
    :param m: 所有 alpha 的数目
    :return:
        j   返回一个不为 i 的随机数，在 0 ~ m 之间的整数值
    """
    j = i
    while j == i:
        j = random.randint(0, m - 1)
    return j


def selectJ(i, oS, Ei):  # this is the second choice -heurstic,and calcs Ej
    """
    selectJ (返回最优的 j 和 Ej)
    内循环的启发式方法
    选择第二个（内循环）alpha的alpha 值
    这里的目标是选择合适的第二个alpha值以保证每次优化中采用最大步长
    该函数的误差与第一个 alpha值 Ei 和下标 i 有关
    :param i:
        具体的下一行
    :param oS:
        optStruct对象
    :param Ei:
        预测结果与真实结果比对，计算误差 Ei
    :return:
        j   随机选出的第 j 行
        Ej  预测结果与真实结果比对，计算误差 Ej
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    #   首先将输入值 Ei 在缓存中设置成为有效的。这里的有效意味着它已经计算好了
    oS.eCache[i] = [1, Ei]
    #   print('oS.eCache[%s]=%s' % (i, oS.eCache[i]))
    #   print('oS.eCache[:, 0].A=%s' % oS.eCache[:, 0].A.T)

    #   返回非0的：行列值
    #   nonzero(oS.eCache[:, 0].A)= (
    #     行： array([ 0,  2,  4,  5,  8, 10, 17, 18, 20, 21, 23, 25, 26, 29, 30, 39, 46,52, 54, 55, 62, 69, 70, 76, 79, 82, 94, 97]),
    #     列： array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0]))

    #   print('nonzero(oS.eCache[:, 0].A)=', nonzero(oS.eCache[:, 0].A))
    #   取行的list
    #   print('nonzero(oS.eCache[:, 0].A)[0]=', nonzero(oS.eCache[:, 0].A)[0])
    #   非零E 值的行的 list 列表，所对应的 alpha 值
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:  # 在所有的值上进行循环，并选择其中使得改变最大的那个值
            if k == i:
                continue  # don't calc for i, waste of time

            #   求 Ek 误差：预测值 - 真实值的差
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                #   选择具有最大步长的 j
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # 如果是第一次循环，则随机选择一个alpha值
        j = selectJrand(i, oS.m)
        #   求 Ek 误差：预测值 - 真实值的差
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    """
    updateEk （计算误差值并存入缓存中）
    在对 alpha 值进行优化之后会用到这个值
    :param oS:
        optStruct对象
    :param k:
        某一行的行号
    """
    #   求误差：预测值 - 真实值的差
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def clipAlpha(aj, H, L):
    """
    clipAlpha(调整 aj 的值，使aj处于 L <= aj <= H)
    :param aj:
        目标值
    :param H:
        最大值
    :param L:
        最小值
    :return:
        aj  目标值
    """
    aj = min(aj, H)
    aj = max(L, aj)
    return aj


def innerL(i, oS):
    """
    innerL 内循环代码
    :param i:
        具体的某一行
    :param oS:
        optStruct
    :return:
        0   找不到最优的值
        1   找到了最优的值，并且 oS.Cache 到缓存中
    """
    #   求 Ek 的误差：预测值 - 真实值的差
    Ei = calcEk(oS, i)
    #   约束条件（KKT条件是解决最优化问题时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值）
    #   0 <= alphas[i] <= C，但由于 0 和 C 是边界值，我们无法进行优化，因为需要增加一个 alphas 和降低一个 alphas。
    #   表示发生错误的概率：labelMat[i] * Ei 如果超出了 toler，才需要优化。至于正负号，我们考虑绝对值就对了。
    #   检验训练样本（xi,yi）是否满足 KKT 条件
    '''
    yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    yi*f(i) == 1 and 0 < alpha < C (on the boundary)
    yi*f(i) <= 1 and alpha = C (between the boundary)
    '''
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        #   选择最大的误差对应的 j 进行优化。效果更明显
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        #   L 和 H 用于将 alphas[j] 调整到 0 ~ C 之间。如果 L == H，就不做任何改变，直接 return 0
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            return 0
        #   eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        #   参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0
        #   计算出一个新的 alphas[j] 的值
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        #   并使用辅助函数，以及 L 和 H 对其进行调整
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        #   更新误差缓存
        updateEk(oS, j)
        #   在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
        #   w= Σ[1~n] ai*yi*xi => b = yi- Σ[1~n] ai*yi(xi*xj)
        #   所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
        #   为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup):
    """
    完整SMO算法外循环，与smoSimple 有些类似，但这里的循环退出条件多一些
    :param dataMatIn:
        数据集
    :param classLabels:
        类别标签
    :param C:
        松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
        控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
        可以通过调节该参数达到不同的结果。
    :param toler:
        容错率
    :param maxIter:
        退出前最大的循环次数
    :param kTup:
        包含核函数信息的元组
    :return:
        b   模型的常量值
        alphas  拉格朗日乘子
    """
    #   创建一个optStruct 对象
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    #   循环遍历：循环 maxIter 次，并且（alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        #   -----   第一种写法 start -----
        #   当 entireSet = true or 非边界 alpha 对没有了；就开始寻找 alpha 对，然后决定是否要进行 else
        if entireSet:
            #   在数据集上遍历所有可能的 alpha
            for i in range(oS.m):
                #   是否存在 alpha对，存在就 +1
                alphaPairsChanged += innerL(i, oS)
                print("fullSet,iter:%d i:%d,pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        #   对已存在 alpha对，选出非边界的 alpha 值，进行优化
        else:
            #   遍历所有的非边界 alpha 值，也就是不在边界 0 或 C 上的值
            nonBoundIs = nonzero((oS.alphs.A > 0) * (oS.alphs.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound,iter:%d i:%d,pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        #   -----   第一种写法 end -----
        #   -----   第二种写法 start -----
        #   if entireSet:
        #       alphaPairsChanged += sum(innerL(i, oS) for i in range(oS.m))
        #   else:
        #       nonBoundIs = nonzero((oS.alphs.A > 0) * (oS.alphs.A < C))[0]
        #       alphaPairsChanged += sum(innerL(i, oS) for i in nonBoundIs)
        #   iter += 1
        #   -----   第二种写法 end -----
        #   如果找到 alpha 对，就优化非边界 alpha 值，否则重新进行寻找，如果寻找一遍/遍历所有的行还是没找到，就退出循环
        if entireSet:
            #   toggle entire set loop
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number:%d" & iter)
    return oS.b, oS.alphs


def calcWs(alphas, dataArr, classLabels):
    """
    基于 alpha 计算 w 值
    :param alphas:
        拉格朗日乘子
    :param dataArr:
        feature 数据集
    :param classLabels:
        目标变量数据集
    :return:
        wc  回归系数
    """
    X = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i].T)
    return w


def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet("/Users/wangjf/Downloads/machinelearninginaction/Ch06/testSetRBF2.txt")
    #   C = 2OO important
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    #   get matrix of only support vectors
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelEval(sVs, dataMat[i, :], ('rbf', k1))

        #   和这个svm-simple类似：fXi = float(multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is:%f" % (float(errorCount) / m))

    dataArr, labelArr = loadDataSet('')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelEval(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelMat[i]):
            errorCount += 1
    print("the test error rate is:%f" % (float(errorCount) / m))


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readlines()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    from os import listdir
    hwLabels = []
    print(dirName)
    #   load the training set
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        #   take off .txt
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('.')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    #   1.导入训练数据
    dataArr, labelArr = loadImages('/Users/wangjf/Downloads/machinelearninginaction/Ch06')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    #   print('there are %d support vectors' %s shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        # 1*m * m*1 = 1*1 单个预测结果
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print('the training error rate is:%f' % (float(errorCount) / m))

    #   2.导入测试数据
    dataArr, labelArr = loadImages('/Users/wangjf/Downloads/machinelearninginaction/Ch06/')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print('the test error rate is: %f' % (float(errorCount) / m))


def plotfig_SVM(xArr, yArr, ws, b, alphas):
    xMat = mat(xArr)
    yMat = mat(yArr)

    #   b 原来是矩阵，先转为数组类型后其数组大小为 (1,1),所以后面加 [0],变为 (1,)
    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #   注意flatten 的用法
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    #   x最大值，最小值根据原数据集 dataArr[:,0] 的大小而定
    x = arange(-1.0, 10.0, 0.1)

    #   根据 x.w + b = 0 得到，其式子展开为 w0.x1 + w1.x2 + b = 0,x2 就说 y 值
    y = (-b - ws[0, 0] * x) / ws[1, 0]
    ax.plot(x, y)
    for i in range(shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

    #   找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()


if __name__ == "__main__":
    testDigits('rbf', 10)
