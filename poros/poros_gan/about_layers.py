# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf


def create_sample_layer(shape, mean=0.0, stdev=1.0):
    """

    :param shape:
    :param mean:
    :param stdev:
    :return:
    """
    with tf.variable_scope("sample"):
        mean = tf.constant(shape=[shape[1]], value=mean)
        stddev = tf.constant(shape=[shape[1]], value=stdev)
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        samples = dist.sample([1], seed=1)

    return samples


def generative_net(x: tf.Tensor, units: list, kernel_initializer=tf.nn.leaky_relu):
    """

    :param x:
    :param units:
    :param kernel_initializer:
    :return:
    """
    with tf.variable_scope("generative_net"):
        inputs = x
        for i, nums in enumerate(units):
            hidden_output = tf.layers.dense(inputs=x, units=units[i], kernel_initializer=kernel_initializer)
            inputs = hidden_output
    return hidden_output
