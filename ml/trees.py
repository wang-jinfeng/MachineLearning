# -*- coding:utf-8 -*-
from math import log
import operator

import pickle

"""
熵（entropy）：
    熵指的是体系的混乱的程度，在不同的学科中也有引申出的更为具体的定义，是各领域十分重要的参量。

信息论（information theory）中的熵（香农熵）：
    是一种信息的度量方式，表示信息的混乱程度，也就是说：信息越有序，信息熵越低。例如：火柴有序放在火柴盒里，熵值很低，相反，熵值很高。

信息增益（information gain）： 
    在划分数据集前后信息发生的变化称为信息增益。
"""


def calcShannonEnt(dataSet):
    #   求list的长度，表示计算参与训练的数据量
    numEntries = len(dataSet)
    #   计算分类标签label出现的次数
    labelCounts = {}
    #   the the number of unique elements and their occurrence
    for featVec in dataSet:
        #   将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = featVec[-1]
        #   为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    #   对于 label 标签的占比，求出 label 标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        #   使用所有类标签的发生频率计算类别出现的概率。
        prob = float(labelCounts[key]) / numEntries
        #   计算香农熵，以 2 为底求对数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


#   按照给定特征划分数据集
def splitDataSet(dataSet, index, value):
    """
    splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)就是依据index列进行分类，如果index列的数据等于 value
    的时候，就要将 index 划分到我们创建的新的数据集中
    :param dataSet:
        数据集，待划分的数据集
    :param index:
        表示每一行的index列，划分数据集的特征
    :param value:
        表示index列对应的value值，需要返回的特征的值
    :return:
        index列为value的数据集[该数据集需要排除index列]
    """
    retDataSet = []
    for featVec in dataSet:
        #   index列为value的数据集[该数据集需要排除index列]
        #   判断index列的值是否为value
        if featVec[index] == value:
            #   chop out index used for splitting
            #   [:index]表示前index行，即若index为2，就是取featVec的前index行
            reducedFeatVec = featVec[:index]
            '''
            请百度查询一下：extend和append的区别
                music_media.append(object) 向列表中添加一个对象object
                music_media.extend(sequence) 把一个序列seq的内容添加到列表中 (跟 += 在list运用类似， music_media += sequence)
            1、使用append的时候，是将object看作一个对象，整体打包添加到music_media对象中。
            2、使用extend的时候，是将sequence看作一个序列，将这个序列和music_media序列合并，并放在其后面。
            music_media = []
            music_media.extend([1,2,3])
            print music_media
            # 结果：
            # [1, 2, 3]
            
            music_media.append([4,5,6])
            print music_media
            # 结果：
            # [1, 2, 3, [4, 5, 6]]
            
            music_media.extend([7,8,9])
            print music_media
            # 结果：
            # [1, 2, 3, [4, 5, 6], 7, 8, 9]
            '''
            reducedFeatVec.extend(featVec[index + 1:])
            #   [index+1:]表示从跳过 index 的 index+1 行，取接下来的数据
            #   收集结果值，index列为 value 的行[该行需要排除index列]
            retDataSet.append(reducedFeatVec)
    return retDataSet


#   按照最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    """
        chooseBestFeatureToSplit(选择最好的特征)
    :param dataSet:
        数据集
    :return:
        bestFeature 最优的特征列
    """
    #   求第一行有多少列的Feature，最后一列是label列
    numFeatures = len(dataSet[0]) - 1
    #   数据集的原始信息熵
    baseEntropy = calcShannonEnt(dataSet)
    #   最优的信息增益值，和最优的Feature编号
    bestInfoGain, bestFeature = 0.0, -1
    #   iterate over all the features
    for i in range(numFeatures):
        #   create a list of all the examples of this feature
        #   获取对应的feature下的所有数据
        featList = [example[i] for example in dataSet]
        #   get a set of unique values
        #   获取剔重合的集合，使用set对list数据进行去重
        uniqueVals = set(featList)
        #   创建一个临时的信息熵
        newEntropy = 0.0
        #   遍历某一列的value集合，计算该列的信息熵
        #   遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            #   计算概率
            prob = len(subDataSet) / float(len(dataSet))
            #   计算信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        #   gain[信息增益]：划分数据集前后的信息变化，获取信息熵最大的值
        #   信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值
        infoGain = baseEntropy - newEntropy
        print("infoGain = ", infoGain, "bestFeature = ", i, baseEntropy, newEntropy)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


"""
    问：上面的 newEntropy 为什么是根据子集计算的呢？
    答：因为我们在根据一个特征计算香农熵的时候，该特征的分类值是相同，这个特征这个分类的香农熵为 0；这就是为什么计算新的香农熵的时候使用的是子集。
"""


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del (labels[bestFeat])
    # 取出最优列，然后它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print 'myTree', value, myTree
    return myTree


#   使用决策树执行分类
def classify(inputTree, featLabels, testVec):
    """
    classify(给输入的节点，进行分类)
    :param inputTree:
        决策树模型
    :param featLabels:
        Feature标签对应的名称
    :param testVec:
        测试输入的数据
    :return:
        classLabel  分类的结果值，需要映射label才能知道名称
    """
    #   获取tree的根节点对应的key值
    firstStr = inputTree.keys()[0]
    #   通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    #   判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    #   测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    #   判断分支是否结束：判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


#   使用pickle模块存储决策树
def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)
