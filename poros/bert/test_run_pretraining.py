# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import unittest
import modeling
import run_pretraining
import tensorflow as tf


class TestRunPretraining(unittest.TestCase):

    def test_get_masked_lm_output(self):
        bert_config = modeling.BertConfig(vocab_size=1000)
        batch_size = 8
        seq_length = 128
        hidden_size = bert_config.hidden_size

        input_tensor = tf.initializers.TruncatedNormal(stddev=0.02)(shape=[batch_size, seq_length, hidden_size])
        output_weights = tf.initializers.TruncatedNormal(stddev=0.02)(shape=[bert_config.vocab_size, hidden_size])
        positions = tf.random.uniform(maxval=2, minval=0, shape=[batch_size, seq_length], dtype=tf.int32)
        label_ids = tf.random.uniform(minval=0, maxval=2, shape=[batch_size, seq_length], dtype=tf.int32)
        label_weights = tf.ones(shape=[batch_size, seq_length])
        loss, per_example_loss, log_probs = run_pretraining.get_masked_lm_output(
            bert_config=bert_config,
            input_tensor=input_tensor,
            output_weights=output_weights,
            positions=positions,
            label_ids=label_ids,
            label_weights=label_weights
        )
        self.assertEqual(loss.shape.as_list(), [])
        self.assertEqual(per_example_loss.shape.as_list(), [batch_size * seq_length])
        self.assertEqual(log_probs.shape.as_list(), [batch_size * seq_length, bert_config.vocab_size])

    def test_get_next_sentence_output(self):
        bert_config = modeling.BertConfig(vocab_size=1000)
        batch_size = 8
        seq_length = 128
        hidden_size = bert_config.hidden_size
        input_tensor = tf.initializers.TruncatedNormal(stddev=0.02)(shape=[batch_size, hidden_size])
        labels = tf.random.uniform(maxval=2, minval=0, shape=[batch_size, 1], dtype=tf.int32)
        loss, per_example_loss, log_probs = run_pretraining.get_next_sentence_output(
            bert_config=bert_config,
            input_tensor=input_tensor,
            labels=labels
        )
        print(log_probs)


