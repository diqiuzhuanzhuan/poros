# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def _parse_function(example_proto, feature_description):
    """Parse the input tf.Example proto using the dictionary above."""
    return tf.io.parse_single_example(example_proto, feature_description)

def create_int_feature(values):
    return _int64_feature(values)

def create_float_feature(values):
    return _float_feature(values)

def create_bytes_feature(value):
    return _bytes_feature(value)

def serialize_example(feature):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    """
    feature is like this:
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }
    """

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def parse_example(example_proto, feature_description):
    """

    :param example_proto:
    :param feature_description:
    :return:
    """
    """
    features_description may like this:
    feature_description = {
        'feature0': tf.FixedLenFeature([], tf.int64, default_value=0),
        'feature1': tf.FixedLenFeature([], tf.int64, default_value=0),
        'feature2': tf.FixedLenFeature([], tf.string, default_value=''),
        'feature3': tf.FixedLenFeature([], tf.float32, default_value=0.0)
    }
    
    """
    # Parse the input tf.Example proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example


def serialize_array(array):
    array = tf.io.serialize_tensor(array).numpy()
    return array


def unserialize_array(bytes_feature, out_type=tf.float64):
    return tf.io.parse_tensor(bytes_feature, out_type=out_type)
