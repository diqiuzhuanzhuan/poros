# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import unittest
from poros.unilmv2.modeling import PositionEmbeddingLayer
from poros.unilmv2.modeling import EmbeddingLookupLayer
import tensorflow as tf
from poros.poros_dataset.about_tensor import get_shape


class ModelingTest(unittest.TestCase):

   def test_embedding_lookup_layer(self):
      embedding_lookup_layer = EmbeddingLookupLayer(vocab_size=100, embedding_size=128)
      input_id = tf.constant([[1, 1, 2, 1]])
      look_up, embedding_table = embedding_lookup_layer(input_id)
      self.assertEqual(get_shape(look_up), [1, 4, 128])
      self.assertEqual(get_shape(embedding_table), [100, 128])

   def test_position_embedding_layer(self):
      position_embedding_layer = PositionEmbeddingLayer(position_size=128, embedding_size=128)
      input_id = tf.constant([[1, 2, 3, 4]])
      look_up, embedding_table = position_embedding_layer(input_id)
      self.assertEqual(get_shape(look_up), [1, 4, 128])
      self.assertEqual(get_shape(embedding_table), [128, 128])


if __name__  == "__main__":
   unittest.main()
