# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import unittest
from poros.unilmv2 import InputEmbeddingLayer
from poros.unilmv2 import (
    Unilmv2Layer,
    MaskLmLayer,
    PseudoMaskLmLayer
)
from poros.unilmv2 import Unilmv2Config
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
        output, embedding_table = input_embedding_layer(inputs=inputs)
        expected_shape = [
            input_ids.shape[0], input_ids.shape[1], self.config.hidden_size
        ]
        self.assertListEqual(output.shape.as_list(), expected_shape)

    def test_unilmv2_layer(self):
        unilmv2_layer = Unilmv2Layer(self.config)
        features = dict()
        features["input_ids"] = tf.constant(
            value=[
                [0, 1, 1, 1, 2, 3, 4, 3, 4, 3, 4, 5]
            ]
        )
        features["attention_mask"] = tf.ones(shape=[features["input_ids"].shape[0], features["input_ids"].shape[0]])
        features["pseudo_masked_index"] = tf.constant(
            value=[
                [7, 8, 2]
            ]
        )
        features["pseudo_masked_sub_list_len"] = tf.constant(
            value=[
                [2, 1]
            ]
        )
        features["output_tokens_positions"] = tf.constant(
            value=[
                [0, 1, 1, 1, 2, 3, 4, 3, 4, 3, 4, 5]
            ]
        )
        batch_size = features["input_ids"].get_shape().as_list()[0]
        output = unilmv2_layer(features)
        self.assertListEqual(output.get_shape().as_list(), [batch_size, self.config.hidden_size])
        sequence_output = unilmv2_layer.get_sequence_output()
        self.assertListEqual(sequence_output.get_shape().as_list(), [batch_size, 12, self.config.hidden_size])

    def test_mask_lm_layer(self):
        mask_lm_layer = MaskLmLayer(config=self.config)
        batch_size, from_seq_length, hidden_size = 2, 128, self.config.hidden_size
        input_tensor = tf.initializers.TruncatedNormal(stddev=0.02)(shape=[batch_size, from_seq_length, hidden_size])
        output_weights = tf.initializers.TruncatedNormal(stddev=0.02)(shape=[self.config.vocab_size, hidden_size])
        positions = tf.constant(
            value=[
                [9, 127, 13, 0, 0, 0],
                [8, 49, 88, 0, 0, 0]
            ]
        )
        label_ids = tf.constant(
            value=[
                [1, 5, 9, 0, 0, 0],
                [3, 7, 12, 0, 0, 0]
            ]
        )
        label_weight = tf.constant(
            value=[
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0]
            ],
            dtype=tf.float32
        )
        loss, per_sample_loss, log_prob = mask_lm_layer(input_tensor, output_weights, positions, label_ids, label_weight)
        print("loss is {}".format(loss))
        print("per sample loss is {}".format(per_sample_loss))
        print("log_prob is {}".format(log_prob))
        self.assertListEqual(log_prob.get_shape().as_list(), [batch_size * positions.get_shape().as_list()[1], self.config.vocab_size])

    def test_pseudo_mask_lm_layer(self):
        pseudo_mask_lm_layer = PseudoMaskLmLayer(config=self.config)
        batch_size , from_seq_length, hidden_size = 2, 128, self.config.hidden_size
        input_tensor = tf.initializers.TruncatedNormal(stddev=0.02)(shape=[batch_size, from_seq_length, hidden_size])
        output_weights = tf.initializers.TruncatedNormal(stddev=0.02)(shape=[self.config.vocab_size, hidden_size])
        positions = tf.constant(
            value=[
                [9, 127, 13, 0, 0, 0],
                [8, 49, 88, 0, 0, 0]
            ]
        )
        label_ids = tf.constant(
            value=[
                [1, 5, 9, 0, 0, 0],
                [3, 7, 12, 0, 0, 0]
            ]
        )
        label_weight = tf.constant(
            value=[
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0]
            ],
            dtype=tf.float32
        )
        loss, per_sample_loss, log_prob = pseudo_mask_lm_layer(input_tensor, output_weights, positions, label_ids, label_weight)
        print("loss is {}".format(loss))
        print("per sample loss is {}".format(per_sample_loss))
        print("log_prob is {}".format(log_prob))
        self.assertListEqual(log_prob.get_shape().as_list(), [batch_size * positions.get_shape().as_list()[1], self.config.vocab_size])


if __name__ == "__main__":
    unittest.main()
