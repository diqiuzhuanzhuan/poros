# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import tensorflow as tf
from poros.poros_gan import about_layers

flag = tf.flags.FLAGS


class Model(object):

    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        about_layers.create_sample_layer(shape=[])



