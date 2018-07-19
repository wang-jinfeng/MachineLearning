#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/7/19 下午5:59
    Author  : wangjf
    File    : CountVectorizer.py
    GitHub  : https://github.com/wjf0627
"""
from sklearn.feature_extraction.text import CountVectorizer

#   语料
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?'
]

#   将文本中的词语转换为词频矩阵
cv = CountVectorizer(min_df=1)
#   计算个词语出现的次数
cv_fit = cv.fit_transform(corpus)

#   set of words (SOW) 词级模型 - 获取词频中所有文本关键词
print('打印所有的特征名称')
print(cv.get_feature_names())

#   bag of words (BOW) 词袋模型
print('打印整个文本矩阵')
print(cv_fit.toarray())
print('打印所有的列相加')
print(cv_fit.toarray().sum(axis=0))
print('打印所有的行相加')
print(cv_fit.toarray().sum(axis=1))
