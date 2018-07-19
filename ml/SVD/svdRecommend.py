#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/7/19 下午4:05
    Author  : wangjf
    File    : svdRecommend.py
    GitHub  : https://github.com/wjf0627
"""

from numpy import *


def loadExData3():
    #   利用 SVD 提高推荐效果，菜肴矩阵
    return [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
            [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]


def loadExData2():
    # 书上代码给的示例矩阵
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def loadExData():
    # 推荐引擎示例矩阵
    return [[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]]

    # # 原矩阵
    # return [[1, 1, 1, 0, 0],
    #         [2, 2, 2, 0, 0],
    #         [1, 1, 1, 0, 0],
    #         [5, 5, 5, 0, 0],
    #         [1, 1, 0, 2, 2],
    #         [0, 0, 0, 3, 3],
    #         [0, 0, 0, 1, 1]]

    # 原矩阵
    # return [[0, -1.6, 0.6],
    #         [0, 1.2, 0.8],
    #         [0, 0, 0],
    #         [0, 0, 0]]


#   相似度计算，假定 inA 和 inB 都是列向量
#   基于欧式距离
def ecloudSim(inA, inB):
    #   如果不存在，该函数返回 1.0，此时两个向量完全相关
    return 1.0 / (1.0 + linalg.norm(inA - inB))


#   pearsSim() 函数会检查是否存在 3 个或更多的点
#   corrcoef 直接计算皮尔逊相关系数，范围 [-1,1]，归一化后 [0,1]
def pearsSim(inA, inB):
    #   如果不存在，该函数返回 1.0，此时两个向量完全相关
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


#   计算余弦相似度，如果夹角为 90 度，相似度为0；如果两个向量的方向相同，相似度为 1.0
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


#   基于物品相似度的推荐引擎
def standEst(dataMat, user, simMeas, item):
    """
    计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相关度，然后进行综合评分
    :param dataMat:
        训练数据集
    :param user:
        用户编号
    :param simMeas:
        相似度计算方法
    :param item:
        未评分的物品编号
    :return:
        ratSimTotal/simTotal    评分（0~5之间的值）
    """
    #   得到数据集中的物品数目
    n = shape(dataMat)[1]
    #   初始化两个评分值
    simTotal = 0.0
    ratSimTotal = 0.0
    #   遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
    for j in range(n):
        userRating = dataMat[user, j]
        #   如果某个物品的评分值为0，则跳过这个物品
        if userRating == 0:
            continue
        #   寻找两个用户都评级的物品
        #   变量 overLap 给出的是两个物品当中已经被评分的那个元素的索引ID
        #   logical_and 计算 x1 和 x2 元素的真值
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        #   如果相似度为 0，则两者没有任何重合元素，终止本次循环
        if len(overLap) == 0:
            similarity = 0
        #   如果存在重合的物品，则基于这些物品重新计算相似度
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])

        print('the %d and %d similarity is：%f' % (item, j, similarity))
        #   相似度会不断增加，每次计算时还考虑相似度和当时用户评分的乘积
        #   similarity  用户相似度，userRating    用户评分
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    #   通过除以所有的评分总和，对上述相似度评分的乘积进行归一化，使得最后评分在 0~5 之间，这些评分用来对预测值进行排序
    else:
        return ratSimTotal / simTotal


#   基于 SVD 的评分估计
#   在recommend() 中，这个函数用于替换对 standEst() 的调用，该函数对给定用户给定物品构建了一个评分估计值
def svdEst(dataMat, user, simMeas, item):
    """
    :param dataMat:
        训练数据集
    :param user:
        用户编号
    :param simMeas:
        相似度计算方法
    :param item:
        未评分的物品编号
    :return:
        ratSimTotal/simTotal    评分（0~5之间）
    """
    #   物品数目
    n = shape(dataMat)[1]
    #   对数据集进行 SVD 分解
    simTotal = 0.0
    ratSimTotal = 0.0
    #   奇异值分解
    #   在 SVD 分解之后，我们只利用包含了 90% 能量值的奇异值，这些奇异值会以 Numpy 数组的形式得以保存
    U, Sigma, VT = linalg.svd(dataMat)
    #   分析 Sigma 的长度取值
    #   analyse_data(Sigma,20)
    #   如果要进行矩阵计算，就必须要用这些奇异值构建出一个对角矩阵
    Sig4 = mat(eye(4) * Sigma[:4])

    #   利用 U 矩阵将物品转换到低维空间中，构建转换后的物品（物品 + 4个主要的特征）
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    print('dataMat', shape(dataMat))
    print('U[:,:4]', shape(U[:, :4]))
    print('Sig4.I', shape(Sig4.I))
    print('VT[:4,:]', shape(VT[:4, :]))
    print('xformedItems', shape(xformedItems))
    #   对于给定的用户，for 循环在用户对应行的元素上进行遍历
    #   这和 standEst() 函数中的 for 循环的目的一样，只不过这里的相似度计算时在低维空间下进行的
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        #   相似度的计算方法也会作为一个参数传递给该函数
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        #   for 循环中加入了一条 print 语句，以便了解相似度计算的进展情况。如果觉得累赘，可以去掉
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        #   对相似度不断累加求和
        simTotal += similarity
        #   对相似度及对应评分值的乘积求和
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        #   计算估计评分
        return ratSimTotal / simTotal


#   recommend() 函数，就是推荐引擎，它默认调用 standEst() 函数，产生了最高的 N 个推荐结果
#   如果不指定 N 的大小，则默认值为3。该函数另外的参数还包括相似度计算方法和估计方法
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    """
    :param dataMat:
        训练数据集
    :param user:
        用户编号
    :param N:
    :param simMeas:
        相似度计算方法
    :param estMethod:
        使用的推荐算法
    :return:
        返回最终 N 个推荐结果
    """
    #   寻找未评级的物品
    #   对给定的用户建立一个未评分的物品列表
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    #   如果不存在未评分物品，那么就退出函数
    if len(unratedItems) == 0:
        return 'you rated everything'
    #   物品的编号和评分值
    itemScores = []
    #   在未评分物品上进行循环
    for item in unratedItems:
        #   获取 item 该物品的平方
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    #   按照评分得分进行逆排序，获取前 N 个未评级物品进行推荐
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


def analyse_data(Sigma, loopNum=20):
    """
    :param Sigma:
        Sigma 的值
    :param loopNum:
        循环次数
    :return:
    """
    #   总方差的集合(总能量值)
    Sig2 = Sigma ** 2
    SigmaSum = sum(Sig2)
    for i in range(loopNum):
        SigmaI = sum(Sig2[:i + 1])
        '''
        根据自己的业务情况，就行处理，设置为 Sigma 次数
        通过保留矩阵 80%~90% 的能量，就可以得到重要的特征并取出噪声
        '''
        print('主成分：%s, 方差占比：%s%%' % (format(i + 1, '2.0f'), format(SigmaI / SigmaSum * 100, '4.2f')))


# 图像压缩函数
# 加载并转换数据
def imgLoadData(filename):
    myl = []
    # 打开文本文件，并从文件以数组方式读入字符
    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    # 矩阵调入后，就可以在屏幕上输出该矩阵
    myMat = mat(myl)
    return myMat


# 打印矩阵
def printMat(inMat, thresh=0.8):
    # 由于矩阵保护了浮点数，因此定义浅色和深色，遍历所有矩阵元素，当元素大于阀值时打印1，否则打印0
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, )
            else:
                print(0, )
        print('')


# 实现图像压缩，允许基于任意给定的奇异值数目来重构图像
def imgCompress(numSV=3, thresh=0.8):
    """imgCompress( )
    Args:
        numSV       Sigma长度
        thresh      判断的阈值
    """
    # 构建一个列表
    myMat = imgLoadData('/Users/wangjf/Downloads/machinelearninginaction/Ch14/0_5.txt')

    print("****original matrix****")
    # 对原始图像进行SVD分解并重构图像e
    printMat(myMat, thresh)

    # 通过Sigma 重新构成SigRecom来实现
    # Sigma是一个对角矩阵，因此需要建立一个全0矩阵，然后将前面的那些奇异值填充到对角线上。
    U, Sigma, VT = linalg.svd(myMat)
    # SigRecon = mat(zeros((numSV, numSV)))
    # for k in range(numSV):
    #     SigRecon[k, k] = Sigma[k]

    # 分析插入的 Sigma 长度
    analyse_data(Sigma, 20)

    SigRecon = mat(eye(numSV) * Sigma[: numSV])
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values *****" % numSV)
    printMat(reconMat, thresh)


if __name__ == "__main__":
    # # 对矩阵进行SVD分解(用python实现SVD)
    # Data = loadExData()
    # print('Data:', Data)
    # U, Sigma, VT = linalg.svd(Data)
    # # 打印Sigma的结果，因为前3个数值比其他的值大了很多，为9.72140007e+00，5.29397912e+00，6.84226362e-01
    # # 后两个值比较小，每台机器输出结果可能有不同可以将这两个值去掉
    # print('U:', U)
    # print('Sigma', Sigma)
    # print('VT:', VT)
    # print('VT:', VT.T)

    # # 重构一个3x3的矩阵Sig3
    # Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # print(U[:, :3] * Sig3 * VT[:3, :])

    """
    # 计算欧氏距离
    myMat = mat(loadExData())
    # print(myMat)
    print(ecludSim(myMat[:, 0], myMat[:, 4]))
    print(ecludSim(myMat[:, 0], myMat[:, 0]))
    # 计算余弦相似度
    print(cosSim(myMat[:, 0], myMat[:, 4]))
    print(cosSim(myMat[:, 0], myMat[:, 0]))
    # 计算皮尔逊相关系数
    print(pearsSim(myMat[:, 0], myMat[:, 4]))
    print(pearsSim(myMat[:, 0], myMat[:, 0]))
    """

    # 计算相似度的方法
    myMat = mat(loadExData3())
    # print(myMat)
    # 计算相似度的第一种方式
    print(recommend(myMat, 1, estMethod=svdEst))
    # 计算相似度的第二种方式
    print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))

    # 默认推荐（菜馆菜肴推荐示例）
    print(recommend(myMat, 2))

    """
    # 利用SVD提高推荐效果
    U, Sigma, VT = la.svd(mat(loadExData2()))
    print(Sigma)                 # 计算矩阵的SVD来了解其需要多少维的特征
    Sig2 = Sigma**2              # 计算需要多少个奇异值能达到总能量的90%
    print(sum(Sig2))             # 计算总能量
    print(sum(Sig2) * 0.9)       # 计算总能量的90%
    print(sum(Sig2[: 2]))        # 计算前两个元素所包含的能量
    print(sum(Sig2[: 3]))        # 两个元素的能量值小于总能量的90%，于是计算前三个元素所包含的能量
    # 该值高于总能量的90%，这就可以了
    """

    # 压缩图片
    # imgCompress(2)
