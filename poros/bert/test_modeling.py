# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import unittest
import tensorflow as tf
import numpy as np
import modeling


class TestModeling(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8

    def tearDown(self) -> None:
        pass

    def test_get_shape_list(self):
        shape = [3, 4, 5, 6]
        truncate_normal = tf.initializers.TruncatedNormal()
        variable = tf.constant(value=truncate_normal(shape))
        self.assertEqual(shape, modeling.get_shape_list(variable, expected_rank=[4]))
        try:
            get_shape = modeling.get_shape_list(variable, expected_rank=[5])
        except ValueError as e:
            get_shape = []
        except Exception as e:
            get_shape = None
        self.assertEqual(get_shape, [])

    def test_embedding_lookup(self):
        batch_size = 8
        seq_length = 256
        vocab_size = 200
        embedding_size = 128
        input_ids = tf.random.uniform(shape=[batch_size, seq_length], minval=0, maxval=vocab_size-1, dtype=tf.int32)
        out, embedding_table = modeling.embedding_lookup(input_ids=input_ids, vocab_size=vocab_size, embedding_size=embedding_size)
        self.assertEqual(out.shape.as_list(), [batch_size, seq_length, embedding_size])
        self.assertEqual(embedding_table.shape.as_list(), [vocab_size, embedding_size])
        print("embedding_table[0:2] is {}".format(embedding_table[0:2]))

    def test_embedding_postprocessor(self):
        pass

    def test_create_attention_mask_from_input_mask(self):
        batch_size = 2
        from_seq_length = 3
        to_seq_length = 4
        from_tensor = tf.constant(value=tf.ones(shape=[batch_size, from_seq_length]))
        to_mask = np.ones(shape=[batch_size, to_seq_length])
        to_mask[0][2] = 0
        to_mask = tf.constant(value=to_mask)
        mask_tensor = modeling.create_attention_mask_from_input_mask(from_tensor, to_mask)
        self.assertEqual(mask_tensor.shape.as_list(), [batch_size, from_seq_length, to_seq_length])
        expected_tensor = np.ones(shape=[batch_size, from_seq_length, to_seq_length])
        expected_tensor[0, :, 2] = 0
        expected_tensor = tf.cast(tf.constant(value=expected_tensor), tf.float32)
        res = tf.math.equal(expected_tensor, mask_tensor).numpy().flatten().all()
        self.assertTrue(res)

    def test_attention_layer(self):
        batch_size = 8
        num_attention_head = 12
        size_per_head = 32
        from_seq_length = 10
        to_seq_length = 10

        attention_mask = tf.zeros(shape=[batch_size, from_seq_length, to_seq_length])
        self.assertEqual(attention_mask.shape, tf.TensorShape([batch_size, from_seq_length, to_seq_length]))
        attention_layer = modeling.AttentionLayer(num_attention_heads=num_attention_head,
                                                  size_per_head=size_per_head,
                                                  batch_size=batch_size)

        q = tf.constant(value=tf.initializers.TruncatedNormal()(shape=[batch_size, from_seq_length, size_per_head]))
        k = tf.constant(value=tf.initializers.TruncatedNormal()(shape=[batch_size, to_seq_length, size_per_head]))
        v = k
        context_layer = attention_layer(q, k, v, attention_mask)
        self.assertEqual(context_layer.shape.as_list(), [batch_size, from_seq_length, num_attention_head * size_per_head])

    def test_transformer_layer(self):
        batch_size = 2
        from_seq_length = 128
        to_seq_length = 128
        width = 768
        transformer_layer = modeling.TransformerLayer(num_hidden_layers=12,
                                  num_attention_heads=12,
                                  intermediate_size=3072)
        input_tensor = tf.constant(value=tf.initializers.TruncatedNormal()(shape=[batch_size, from_seq_length, width]))
        attention_mask = modeling.create_attention_mask_from_input_mask(input_tensor, tf.zeros(shape=[batch_size, to_seq_length]))
        output_tensor = transformer_layer(input_tensor=input_tensor, attention_mask=attention_mask)
        print("output_tensor shape is {}".format(output_tensor.shape))


if __name__ == '__main__':
    unittest.main()
