# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import unittest
import tensorflow as tf
import numpy as np
from poros.bert import modeling


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

    def test_layer_norm_and_dropout(self):
        input_tensor = tf.random.uniform(shape=[8, 128])
        drop_prob = 0.1
        output_tensor = modeling.layer_norm_and_dropout(input_tensor, drop_prob)
        print(input_tensor)
        print(output_tensor)
        self.assertEqual(input_tensor.shape.as_list(), output_tensor.shape.as_list())

    def test_embedding_lookup(self):
        batch_size = 8
        seq_length = 256
        vocab_size = 200
        embedding_size = 128
        input_ids = tf.random.uniform(shape=[batch_size, seq_length], minval=0, maxval=vocab_size-1, dtype=tf.int32)
        out, embedding_table = modeling.embedding_lookup(input_ids=input_ids, vocab_size=vocab_size, embedding_size=embedding_size)
        self.assertEqual(out.shape.as_list(), [batch_size, seq_length, embedding_size])
        self.assertEqual(embedding_table.shape.as_list(), [vocab_size, embedding_size])
        print("embedding_table[0] is {}".format(embedding_table[0]))

    def test_embedding_lookup_layer(self):
        vocab_size = 1000
        embedding_size = 128
        batch_size = 8
        seq_length = 12
        use_one_hot_embeddings = False
        embedding_lookup_layer = modeling.EmbeddingLookupLayer(vocab_size=vocab_size, embedding_size=embedding_size)
        print(embedding_lookup_layer.trainable_variables)
        input_ids = tf.random.uniform(maxval=vocab_size, minval=0, shape=[batch_size, seq_length], dtype=tf.int32)
        output, embedding_table = embedding_lookup_layer(input_ids, use_one_hot_embeddings)
        self.assertEqual(output.shape.as_list(), [batch_size, seq_length, embedding_size])
        self.assertEqual(embedding_table.shape.as_list(), [vocab_size, embedding_size])

    def test_embedding_postprocessor_layer(self):
        embedding_size = 768
        max_position_embedding = 512
        embedding_postprocessor_layer = modeling.EmbeddingPostprocessorLayer(
            use_token_type=True,
            embedding_size=embedding_size,
            max_position_embeddings=max_position_embedding
        )
        print(embedding_postprocessor_layer.trainable_variables)
        batch_size = 8
        seq_length = 128
        input_tensor = tf.random.uniform(shape=[batch_size, seq_length, embedding_size])
        token_tensor = tf.random.uniform(shape=[batch_size, seq_length], maxval=1, minval=0, dtype=tf.int32)
        output = embedding_postprocessor_layer(
            input_tensor, token_tensor
        )
        self.assertEqual(output.shape.as_list(), input_tensor.shape.as_list())

    def test_embedding_postprocessor(self):
        batch_size = 8
        seq_length = 128
        embedding_size = 128
        input_tensor = tf.initializers.TruncatedNormal(stddev=0.2)(shape=[batch_size, seq_length, embedding_size])
        out_tensor = modeling.embedding_postprocessor(input_tensor=input_tensor)
        self.assertEqual(out_tensor.shape.as_list(), [batch_size, seq_length, embedding_size])

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
        size_per_head = 64
        from_seq_length = 10
        to_seq_length = 10

        attention_mask = tf.zeros(shape=[batch_size, from_seq_length, to_seq_length])
        self.assertEqual(attention_mask.shape, tf.TensorShape([batch_size, from_seq_length, to_seq_length]))
        attention_layer = modeling.AttentionLayer(
            num_attention_heads=num_attention_head,
            size_per_head=size_per_head,
        )

        q = tf.constant(value=tf.initializers.TruncatedNormal()(
            shape=[batch_size, from_seq_length, num_attention_head *size_per_head]))
        k = tf.constant(value=tf.initializers.TruncatedNormal()(
            shape=[batch_size, to_seq_length, num_attention_head * size_per_head]))
        v = k
        print(attention_layer.wq.trainable_variables)
        print(attention_layer.trainable_variables)
        context_layer = attention_layer(q, k, v, attention_mask)
        self.assertEqual(context_layer.shape.as_list(), [batch_size, from_seq_length, num_attention_head * size_per_head])

    def test_transformer_layer(self):
        batch_size = 2
        from_seq_length = 128
        to_seq_length = 128
        width = 768
        transformer_layer = modeling.TransformerLayer(
            hidden_size=width,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072)
        print(transformer_layer.trainable_variables)
        input_tensor = tf.constant(value=tf.initializers.TruncatedNormal()(shape=[batch_size, from_seq_length, width]))
        attention_mask = modeling.create_attention_mask_from_input_mask(input_tensor, tf.zeros(shape=[batch_size, to_seq_length]))
        output_tensor = transformer_layer(input_tensor, attention_mask)
        print("output_tensor shape is {}".format(output_tensor.shape))

    def test_bert_layer(self):
        vocab_size = 211128
        hidden_size = 768
        type_vocab_size = 2
        bert_config = modeling.BertConfig(vocab_size=vocab_size,hidden_size=hidden_size, type_vocab_size=type_vocab_size)
        batch_size = 8
        seq_length = 128
        bert_layer = modeling.BertLayer(config=bert_config, is_training=True)
        #print(bert_layer.trainable_variables)
        for ele in bert_layer.trainable_variables:
            print(ele.shape, ele.name)
        input_ids = tf.random.uniform(shape=[batch_size, seq_length], minval=0, maxval=vocab_size-1, dtype=tf.int32)
        features = {
            "input_ids": input_ids,
            "input_mask": None,
            "token_type_ids": None,
            "scope": "bert"
        }
        output_tensor = bert_layer(input_ids, input_mask=None, token_type_ids=None)
        self.assertEqual(output_tensor.shape.as_list(), [batch_size, hidden_size])


if __name__ == '__main__':
    unittest.main()
