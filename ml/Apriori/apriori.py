#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/7/5 下午5:26
    Author  : wangjf
    File    : apriori.py
    GitHub  : https://github.com/wjf0627
"""

#   加载数据集
from time import sleep

from numpy import *
from votesmart import votesmart


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


#   创建集合 C1。即对 dataSet 进行去重，放入 list 中，然后转换为所有的元素为 frozenset
def createC1(dataSet):
    """
    createC1 (创建集合 C1)
    :param dataSet:
        原始数据集
    :return:
        frozenset 返回一个 frozenset 格式的 list
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                #   遍历所有的元素，如果不在 C1 出现过，那么就 append
                C1.append([item])
        #   对数组进行 '从小到大' 的排序
        #   print('sort 前 = ', C1)
    C1.sort()
    #   frozenset 表示冻结的 set 集合，元素无改变，可以把它当字典的 key 来使用
    #   print('sort 后 = ', C1)
    #   print('frozenset = ', map(frozenset, C1))
    return map(frozenset, C1)


#   计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度 (minSupport) 的数据
def scanD(D, Ck, minSupport):
    """
    scanD (计算候选数据集 Ck 在数据集 D 中的支持度，并返回支持度大于最小支持度 minSupport 的数据)
    :param D:
        数据集
    :param Ck:
        候选数据集
    :param minSupport:
        最小支持度
    :return:
        retList 支持度大于 minSupport 的集合
    """
    #   ssCnt 临时存放候选数据集 Ck 的频率。例如：a->10,b->5,c->8
    ssCnt = {}
    for tid in D:
        for can in Ck:
            #   s.issubset(t)   测试是否 s 中的每一个元素都在 t 中
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(D.__len__())  # 数据集 D 的数量
    retList = []
    supportData = {}
    for key in ssCnt:
        #   支持度 = 候选项(key) 出现的次数 / 所有数据集的数量
        support = ssCnt[key] / numItems
        if support >= minSupport:
            #   在 retList 的首位插入元素，只存储支持度满足频繁项集的值
            retList.insert(0, key)
        #   存储所有的候选项(key) 和对应的支持度(support)
        supportData[key] = support
    return retList, supportData


#   输入频繁项集列表 Lk 与返回的元素个数 k，然后输出所有可能的候选项集 Ck
def aprioriGen(Lk, k):
    """
    aprioriGen 输入频繁项集列表 Lk 与返回的元素个数 k，然后输出候选项集 Ck
    例如：以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1},{0,2},{1,2}。以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
    仅需要计算一次，不需要将所有的结果计算出来，然后进行去重操作。这是一个更高效的算法
    :param Lk:
        频繁项集列表
    :param k:
        返回的项集元素个数(若元素的前 k - 2 相同，就进行合并)
    :return:
        retList 元素两两合并的数据集
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[: k - 2]
            L2 = list(Lk[j])[: k - 2]
            L1.sort()
            L2.sort()
            #   第一次 L1,L2 为空，元素直接进行合并，返回元素两两合并的数据集
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


#   找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集
def apriori(dataSet, minSupport=0.5):
    """
    apriori 首先构建集合 C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。那么满足最小支持度要求的项集构成集合 L1。
    然后 L1 中的元素相互组合成 C2，C2 再进一步过滤变成 L2，然后以此类推，直到 CN 的长度为 0 时结束，即可找出所有频繁项集的支持度
    :param dataSet:
        原始数据集
    :param minSupport:
        支持度的阈值
    :return:
        L   频繁项集的全集
        supportData 所有元素和支持度的全集
    """
    #   C1 即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
    C1 = createC1(dataSet)
    #   对每一行进行 set 转换，然后存放到集合中
    D = list(map(frozenset, dataSet))
    #   计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
    L1, supportData = scanD(D, list(C1), minSupport)
    L = [L1]
    k = 2
    #   判断 L 的第 k-2 项的数据长度是否 > 0。
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)

        Lk, supK = scanD(D, Ck, minSupport)  # 计算候选数据集 Ck 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
        #   保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
        supportData.update(supK)
        if len(Lk) == 0:
            break
        #   Lk 表示满足频繁子项的集合，元素在增加。
        #   l = [[set(1),set(2),set(3)]]
        #   l = [[set(1),set(2),set(3)],[set(1,2),set(2,3)]]
        L.append(Lk)
        k += 1
    return L, supportData


#   计算可信度(confidence)
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    对两个元素的频繁项，计算可信度，例如：{1,2}/{1} 或者 {1,2}/{2} 看是否满足条件
    :param freSet:
        频繁项集中的元素，例如：frozenset([1,3])
    :param H:
        频繁项集中的元素的集合，例如：[frozenset([1]),frozenset([3])]
    :param supportData:
        所有元素的支持度的字典
    :param brl:
        关联规则列表的空数组
    :param minConf:
        最小可信度
    :return:
        prunedH 记录 可信度大于阈值的集合
    """
    #   记录可信度大于最小可信度（minConf）的集合
    prunedH = []
    #   假设 freqSet = frozenset([1,3]),H = [frozenset([1]),frozenset([3])]，那么现在需要求出
    #   frozenset([1]) -> frozenset([3]) 的可信度和 frozenset([3]) -> frozenset([1]) 的可信度
    for conseq in H:
        #   支持度定义：a -> b = support(a | b)/support(a)。假设 freqSet = frozenset([1,3])，conseq = [frozenset([1])]，那么
        #   frozenset([1]) 至 frozenset([3]) 的可信度为 = support(a | b)/support(a)
        #   = supportData[freqSet]/supportData[freqSet - conseq]
        #   = supportData[frozenset([1,3])] / supportData[frozenset([1])]
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            #   只要买了 freqSet - conseq 集合，一定会买 conseq 集合 (freqSet - conseq 集合和 conseq 集合 是全集)
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


#   递归计算频繁项集的规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    :param freqSet:
        频繁项集中的元素，例如：frozenset([2,3,5])
    :param H:
        频繁项集中的元素的集合，例如：[frozenset([2]),frozenset([3]),frozenset([5])]
    :param supportData:
        所有元素的支持度的字典
    :param brl:
        关联规则列表的数组
    :param minConf:
        最小可信度
    :return:
    """
    #   H[0] 是 freqSet 的元素组合的第一个元素，并且 H 中所有元素的长度都一样，长度由 aprioriGen(H,m+1) 这里的 m + 1 来控制
    #   该函数递归时，H[0] 的长度从 1 开始增长 1，2，3...
    #   假设 freqSet = frozenset([2,3,5]),H = [frozenset([2,3,5]),frozenset([2,3,5])]
    #   那么 m = len(H[0]) 的递归的值依次为 1 2
    #   在 m = 2 时，跳出该递归。假设再递归一次，那么 H[0] = frozenset([2,3,5]),freqSet = frozenset([2,3,5]),没必要再计算 freqSet 与 H[0] 的关联规则了
    m = len(H[0])
    if len(freqSet) > (m + 1):
        #   生成 m + 1 个长度的所有可能的 H 中的组合，假设 H = [frozenset([2]),frozenset([3]),frozenset([5])]
        #   第一次递归调用时生成 [frozenset([2,3]),frozenset([2,5]),frozenset([3,5])]
        #   第二次。。。没有第二次，递归条件判断时已经退出了
        Hmp1 = apriori(H, m + 1)
        #   返回可信度大于最小可信度的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        print('Hmp1=', Hmp1)
        print('len(Hmp1) = ', len(Hmp1), 'len(freqSet) = ', len(freqSet))
        #   计算可信度后，还有数据大于最小可信度的话，那么继续递归调用，否则跳出递归
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


#   生成关联规则
def generateRules(L, supportData, minConf=0.7):
    """
    :param L:
        频繁项集列表
    :param supportData:
        频繁项集支持度的字典
    :param minConf:
        最小置信度
    :return:
        bigRuleList 可信度规则列表(关于 (A->B + 置信度) 3个字段的组合)
    """
    bigRuleList = []
    #   假设 L = [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])], [frozenset([1, 3]), frozenset([2, 5]), frozenset([2, 3]), frozenset([3, 5])], [frozenset([2, 3, 5])]]
    for i in range(1, len(L)):
        #   获取频繁项集中每个组合的所有元素
        for freqSet in L[i]:
            #   假设：freqSet = frozenset([1,3]),H1 = [frozenset([1]),frozenset([3])]
            #   组合总的元素并遍历子元素，并转化为 frozenset 集合，再存放到 list 列表中
            H1 = [frozenset([item]) for item in freqSet]
            #   2 个的组合，走 else，2 个 以上的组合，走 if
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def getActionId():
    votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
    actionIdList = []
    billTitleList = []
    fr = open('/Users/wangjf/Downloads/machinelearninginaction/Ch11/recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum, "")
            for action in billDetail.actions:
                if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print('problem getting bill %d' % billNum)
        sleep(1)
    return actionIdList, billTitleList


def getTransList(actionIdList, billTitleList):
    itemMeaning = ['Republican', 'Democratic']
    for billTitle in billTitleList:
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print('problem getting actionId: %' % actionId)
        voteCount += 2
    return transDict, itemMeaning


def testApriori():
    #   加载测试数据集
    dataSet = loadDataSet()
    print('dataSet: ', dataSet)

    #   Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori(dataSet, minSupport=0.7)
    print('L(0.7): ', L1)
    print('supportData(0.7): ', supportData1)

    print('->->->->->->->->->->->->->->->->->->->->->->->->->->->')

    #   Apriori 算法生成频繁项集以及它们的支持度
    L2, supportData2 = apriori(dataSet, minSupport=0.5)
    print('L(0.5): ', L2)
    print('supportData(0.2): ', supportData2)


def testGenerateRules():
    #   加载测试数据集
    dataSet = loadDataSet()
    print('dataSet: ', dataSet)

    #   Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori(dataSet, minSupport=0.5)
    print('L(0.7): ', L1)
    print('supportData(0.7): ', supportData1)

    #   生成关联规则
    rules = generateRules(L1, supportData1, minConf=0.5)
    print('rules: ', rules)


def main():
    #   测试 Apriori 算法
    #   testApriori()

    #   生成关联规则
    #   testGenerateRules()

    # # 得到全集的数据

    """
    dataSet = [line.split() for line in
               open("/Users/wangjf/Downloads/machinelearninginaction/Ch11/mushroom.dat").readlines()]
    L, supportData = apriori(dataSet, minSupport=0.3)
    # # 2表示毒蘑菇，1表示可食用的蘑菇
    # # 找出关于2的频繁子项出来，就知道如果是毒蘑菇，那么出现频繁的也可能是毒蘑菇
    for item in L[1]:
        if item.intersection('2'):
            print(item)
    
    for item in L[2]:
        if item.intersection('2'):
            print(item)
    """
    #   dataSet = loadDataSet()
    #   C1 = createC1(dataSet)
    #   print(C1)
    actionIdList, billTitles = getActionId()
    print(actionIdList)
    print(billTitles)


if __name__ == '__main__':
    main()
