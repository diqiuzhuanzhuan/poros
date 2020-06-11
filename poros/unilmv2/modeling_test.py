# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import unittest
from poros.unilmv2.modeling import InputEmbeddingLayer
from poros.unilmv2.modeling import Unilmv2Layer
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

    def test_create_attention_mask(self):
        expected_matrix = tf.constant(
            value=[
                [
                    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                ]
            ]
        )
        input_ids = tf.constant(
            value=[
                [0, 1, 1, 1, 2, 3, 4, 3, 4, 3, 4, 5]
            ]
        )

        pseudo_index = tf.constant(
            value=[
                [7, 8, 2]
            ]
        )
        pseudo_len = tf.constant(
            value=[
                [2, 1]
            ]
        )
        unilmv2_layer = Unilmv2Layer(Unilmv2Config(vocab_size=100))
        mask_matrix = unilmv2_layer.create_attention_mask(input_ids, pseudo_index, pseudo_len)
        self.assertListEqual(expected_matrix.numpy().tolist(), mask_matrix.numpy().tolist())


if __name__ == "__main__":
    unittest.main()
