#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Time    : 2018/7/24 下午3:03
    Author  : wangjf
    File    : Demo-CBoW.py
    GitHub  : https://github.com/wjf0627
"""
import torch
import torch.nn as nn
import torch.autograd as autograd

#   torch.manual_seed(1)
#   word_to_ix = {"hello": 2, "world": 1}
#   2 表示有 2 个词，5 表示 5 维度，其实也就是一个 2 * 5 的矩阵
#   embeds = nn.Embedding(2, 5)
#   lookup_tensor = torch.CudaLongTensorBase([word_to_ix["hello"]])
#   hello_embed = embeds(autograd.Variable(lookup_tensor))
#   print(hello_embed)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

#   By deriving a set from `raw_text`,we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):
    def __init__(self):
        pass

    def forward(self, *input):
        pass


#   create your model and train.  here are some functions to help you make
#   the data ready for use by your module
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


make_context_vector(data[0][0], word_to_ix)
