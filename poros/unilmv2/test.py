# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


import tensorflow as tf

inputs = [position_embeding + word_embedding + segmeng_id]

head = inputs * w

q = head_q
k = head_k
v = head_v
a = attention(q, k, v)
t = concate([a, ...])
forward(t)
output = layer()

q = output * wq
k = output * wk
k = output * wv
attention



