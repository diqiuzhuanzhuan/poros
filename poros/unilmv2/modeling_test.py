# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import unittest
from poros.unilmv2.modeling import InputEmbeddingLayer
from poros.unilmv2.config import Unilmv2Config
import tensorflow as tf
import os


class ModelingTest(unittest.TestCase):

   @classmethod
   def setUpClass(cls) -> None:
      cls.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
      cls.config_file = os.path.join(cls.test_data_path, "bert_config.json")
      cls.config = Unilmv2Config.from_json_file(cls.config_file)

   def test_input_embedding_layer(self):
      input_embedding_layer = InputEmbeddingLayer(config=self.config)
      input_ids = tf.constant([
         [1, 2, 3, 4],
         [1, 2, 7, 9]
                               ])
      position_ids = tf.constant([
         [1, 2, 3, 4],
         [1, 2, 3, 4]
      ])
      token_type_ids = tf.constant([
         [0, 0, 1, 1],
         [0, 0, 0, 1]
      ])
      inputs = {
         "input_ids": input_ids,
         "position_ids": position_ids,
         "token_type_ids": token_type_ids
      }
      output = input_embedding_layer(inputs=inputs)
      expected_shape = [
         input_ids.shape[0], input_ids.shape[1], self.config.hidden_size
      ]
      self.assertListEqual(output.shape.as_list(), expected_shape)


if __name__  == "__main__":
   unittest.main()
