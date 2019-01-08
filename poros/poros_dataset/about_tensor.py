# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import tensorflow as tf


def get_shape(tensor):
    """

    :param tensor: a tensor defined by tensorflow
    :return: static shape when it can
    """
    static_shape = tensor.get_shape().as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [
        s[1] if s[0] is None else s[0] for s in zip(static_shape, dynamic_shape)
    ]
    return dims
