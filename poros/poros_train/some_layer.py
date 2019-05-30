# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
from poros.poros_dataset import about_tensor


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def dot_product_attention(q, k, v, scale=True, bias=None, mask=None):
    """

    :param q:
    :param k:
    :param v:
    :param bias:
    :param dropout_rate:
    :param type:
    :return:
    """
    with tf.variable_scope("dot_product_attention"):
        logits = tf.matmul(q, k, transpose_b=True)

        if bias:
            logits += bias
        if scale:
            logits = logits / tf.sqrt(tf.cast(about_tensor.get_shape(v)[-1], tf.float32))
        if mask:
            logits += mask * 1e-9
        weights = tf.nn.softmax(logits, name="attention_weights")

    return tf.matmul(weights, v)



