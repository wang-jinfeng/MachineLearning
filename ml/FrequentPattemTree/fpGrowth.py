#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/7/13 下午5:02
    Author  : wangjf
    File    : fpGrowth.py
    GitHub  : https://github.com/wjf0627
"""


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        """
        对 count 变量增加给定值
        :param numOccur:
        :return:
        """
        self.count += numOccur

    def disp(self, ind=1):
        """
        用于将树以文本形式展现
        :param ind:
        :return:
        """
        print(' ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def loadSimpDat():
    simpDat = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        if frozenset(trans) not in retDict.keys():
            retDict[frozenset(trans)] = 1
        else:
            retDict[frozenset(trans)] += 1
    return retDict


def updateHeader(nodeToTest, targetNode):
    """
    更新头指针，建立相同元素之间的关系。例如：左边的 r 指向右边的 r 值，就是后出现的相同元素 指向 已出现的元素
    从头指针 nodeLink 开始，一直沿着 nodeLink 直到到达链表末尾。
    性能：如果链表很长可能会遇到迭代调用的次数限制
    :param nodeToTest:
        满足 minSup {所有的元素+(value,treeNode)}
    :param targetNode:
        Tree 对象的子节点
    :return:
    """
    #   建立相同元素之间的关系，例如：左边的 r 指向 右边的 r 值
    while nodeToTest.nodeLink is not None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = treeNode


def updateTree(items, inTree, headerTable, count):
    """
    更新 FP-tree，第二次遍历

    针对每一行的数据
    最大的key，添加
    :param items:
        满足 minSup 排序后的元素 key 的数组(大到小的排序)
    :param inTree:
        空的 Tree 对象
    :param headerTable:
        满足 minSup {所有的元素+(value,treeNode)}
    :param count:
        原数据集中每一组 key 出现的次数
    :return:
    """
    #   取出 元素 出现次数最高的
    #   如果该元素在 inTree.children 这个字典中，就进行累加
    #   如果该元素不存在，就 inTree.children 字典中新增 key，value 为初始化的 treeNode 对象
    if items[0] in inTree.children:
        #   更新最大元素，对应的 treeNode 对象的count进行叠加
        inTree.children[items[0].inc(count)]
    else:
        #   如果不存在子节点，我们为该 inTree 添加子节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        #   如果满足 minSup 的 dist 字典的 value 值第二位为 null，我们就设置该元素为 本节点对应的 tree 节点
        #   如果元素第二位不为 null，我们就更新 header 节点
        if headerTable[items[0]][1] is None:
            headerTable[items[0][1]] = inTree.children[items[0]]
        else:
            #   本质上是修改 headerTable 的 key 对应的 Tree 的 nodeLink 值
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        #   递归的调用，在 items[0] 的基础上，添加 items[0][1] 做子节点，count 只要循环的进行累加和而已，统计出节点的最后的统计值
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)

    def createTree(dataSet, minSup=1):
        """
        生成 FP-tree
        :param dataSet:
            dist{行:出现次数}的样本数据
        :param minSup:
            最小的支持度
        :return:
            retTree FP-tree
            headerTable 满足 minSup {所有的元素+(value,treeNode)}
        """
        #   支持度 >= minSup 的 dist {所有元素：出现的次数}
        headerTable = {}
        #   循环 dist{行:出现次数} 的样本数据
        for trans in dataSet:
            #   对所有的行进行循环，得到行里面的所有元素
            #   统计每一行中，每个元素出现的总次数
            for item in trans:
                #   例如：{'ababa',3} count(a) = 3+3+3 = 9，count(b) = 3+3 = 6
                headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
        #   删除 headerTable 中，元素次数 < 最小支持度的元素
        for k in list(headerTable.keys()):
            if headerTable[k] < minSup:
                del (headerTable[k])

        #   满足 minSup:set(各元素集合)
