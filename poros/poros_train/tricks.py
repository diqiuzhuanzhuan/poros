# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import os
import tensorflow as tf

PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]


# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device

    return _assign


def get_strategy(name: str):
    print("Tensorflow version " + tf.__version__)
    if name.lower() == "tpu":
        tpu = 'grpc://' + os.environ['COLAB_TPU_ADDR']

        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu)  # TPU detection
            print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        except ValueError:
            raise BaseException(
                'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
        return tpu_strategy
    else:
        gpu_strategy = tf.distribute.MirroredStrategy()
        return gpu_strategy


