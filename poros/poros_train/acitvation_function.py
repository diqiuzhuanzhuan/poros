# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
import numpy as np
import six


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "relu6":
        return tf.nn.relu6
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    elif act == "selu":
        return tf.nn.selu
    elif act == "elu":
        return tf.nn.elu
    elif act == "leaky_relu":
        return tf.nn.leaky_relu
    elif act == "swish":
        return swish
    else:
        raise ValueError("Unsupported activation: %s" % act)


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    return x * tf.sigmoid(x)

