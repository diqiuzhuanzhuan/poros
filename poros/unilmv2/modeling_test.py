# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import unittest
from unilmv2.modeling import PositionEmbeddingLayer
import tensorflow as tf
from poros_dataset.about_tensor import get_shape


class ModelingTest(unittest.TestCase):

   def test_embedding_lookup_layer(self):
      pass

   def test_position_embedding_layer(self):
      position_embedding_layer = PositionEmbeddingLayer(position_size=128, embedding_size=128)
      input_id = tf.constant([[1, 2, 3, 4]])
      look_up, embedding_table = position_embedding_layer(input_id)
      self.assertEqual(get_shape(look_up), [1, 4, 128])
      self.assertEqual(get_shape(embedding_table), [128, 128])

