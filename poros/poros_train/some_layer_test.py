# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
from poros.poros_train import some_layer
from poros_dataset.about_tensor import get_shape
import numpy as np


class SomeLayerTest(tf.test.TestCase):

    def test_dropout(self):
        input_tensor = tf.random.truncated_normal(shape=[1, 9], stddev=0.02)
        input_tensor = tf.reshape(input_tensor, [-1, 3])
        print(input_tensor.get_shape().rank)
        some_layer.dropout(input_tensor, 0.1)

    def test_dot_product_attention(self):

        query = tf.Variable(initial_value=[
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ])

        out = some_layer.dot_product_attention(q=query, k=query, v=query, scale=True, bias=None)
        out_ = tf.constant([
            [2.85093709, 2.85093709, 2.85093709, 2.85093709],
            [2.98136107, 2.98136107, 2.98136107, 2.98136107],
            [2.99751513, 2.99751513, 2.99751513, 2.99751513]
        ], dtype=tf.float32
        )
        self.assertAllClose(out, out_)

    def test_embedding_lookup_layer(self):
        embedding_lookup_layer = some_layer.EmbeddingLookupLayer(vocab_size=100, embedding_size=128)
        input_id = tf.constant([[1, 1, 2, 1]])
        look_up, embedding_table = embedding_lookup_layer(input_id)
        self.assertEqual(get_shape(look_up), [1, 4, 128])
        self.assertEqual(get_shape(embedding_table), [100, 128])

    def test_position_embedding_layer(self):
        position_embedding_layer = some_layer.PositionEmbeddingLayer(position_size=128, embedding_size=128)
        input_id = tf.constant([[1, 2, 3, 4]])
        look_up, embedding_table = position_embedding_layer(input_id)
        self.assertEqual(get_shape(look_up), [1, 4, 128])
        self.assertEqual(get_shape(embedding_table), [128, 128])

    def test_attention_layer(self):
        attention_layer = some_layer.AttentionLayer(size_per_head=768)
        q = tf.random.truncated_normal(shape=[1, 10, 768])
        k = q
        v = q
        numpy_mask = np.zeros(shape=[1, 10, 10])
        numpy_mask[:, :, 0] = 1
        attention_mask = tf.constant(value=numpy_mask)

        res = attention_layer(q, k, v, attention_mask=attention_mask)
        print(res)

    def test_rpr_attention_layer(self):
        clip_k = 5
        attention_layer = some_layer.AttentionLayerWithRPR(num_attention_heads=12, size_per_head=64, clip_k=5)
        rpr_matrix = some_layer.create_rpr_matrix(clip_k, 10, 10)
        q = tf.random.truncated_normal(shape=[1, 10, 768])
        k = q
        v = q
        res = attention_layer(q, k, v, rpr_matrix=rpr_matrix)
        print(res)

    def test_create_rpr_matrix(self):
        k = 5
        from_length = 11
        to_length = 10
        matrix = some_layer.create_rpr_matrix(k, from_length, to_length)
        print(matrix)
        for i in range(from_length):
            for j in range(to_length):
                if j - i < -k:
                    self.assertEqual(matrix[i, j], 0)
                elif j - i > k:
                    self.assertEqual(matrix[i, j], 2*k)
                else:
                    self.assertEqual(matrix[i, j], j - i + k)


    def test_mmoe_layer(self):
        mmoe_layer = some_layer.MMoELayer(units=10, num_tasks=5, num_experts=15)
        x = tf.random.truncated_normal(shape=[8, 10], mean=0, stddev=0.01)
        y = mmoe_layer(x)
        print(y)


if __name__ == "__main__":
    tf.test.main()
