# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf


def create_sample_layer(shape):
    """
    :param shape:
    :return:
    """
    with tf.variable_scope("sample"):
        mean = tf.constant(shape=[shape[1]], value=0.0)
        stddev = tf.constant(shape=[shape[1]], value=1.0)
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        samples = dist.sample([1], seed=1)

    return samples
