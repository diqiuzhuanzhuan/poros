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
        hidden_size = bert_config.hidden_size
        input_tensor = tf.initializers.TruncatedNormal(stddev=0.02)(shape=[batch_size, hidden_size])
        labels = tf.random.uniform(maxval=2, minval=0, shape=[batch_size, 1], dtype=tf.int32)
        loss, per_example_loss, log_probs = run_pretraining.get_next_sentence_output(
            bert_config=bert_config,
            input_tensor=input_tensor,
            labels=labels
        )
        self.assertEqual(loss.shape.as_list(), [])
        self.assertEqual(per_example_loss.shape.as_list(), [batch_size])
        self.assertEqual(log_probs.shape.as_list(), [batch_size, 2])

    def test_bert_pretrain_model(self):
        bert_config = modeling.BertConfig(vocab_size=1000)
        max_seq_length = 128
        max_predictions_per_seq = 20
        batch_size = 8
        features = {
            "input_ids": tf.random.uniform(maxval=100, minval=0, shape=[batch_size, max_seq_length],dtype=tf.int32),
            "input_mask": tf.random.uniform(maxval=2, minval=0, shape=[batch_size, max_seq_length], dtype=tf.int32),
            "segment_ids": tf.concat([tf.zeros(shape=[batch_size, 60], dtype=tf.int32),
                                      tf.zeros(shape=[batch_size, max_seq_length - 60], dtype=tf.int32)], axis=-1),
            "masked_lm_positions": tf.random.uniform(
                maxval=max_seq_length, minval=0, shape=[batch_size, max_predictions_per_seq], dtype=tf.int32),
            "masked_lm_ids": tf.random.uniform(
                maxval=max_seq_length, minval=0, shape=[batch_size, max_predictions_per_seq], dtype=tf.int32),
            "masked_lm_weights": tf.initializers.TruncatedNormal(stddev=0.02)(shape=[batch_size, max_predictions_per_seq]),
            "next_sentence_labels": tf.random.uniform(maxval=2, minval=0, shape=[batch_size, 1], dtype=tf.int32)
        }
        bert_pretrain_model = run_pretraining.BertPretrainModel(
            config=bert_config,
            is_training=True,
            init_checkpoint="../bert_model/data/chinese_L-12_H-768_A-12"
        )
        bert_pretrain_model(features=features)
        bert_pretrain_model(features=features)
        bert_pretrain_model(features=features)
        bert_pretrain_model(features=features)


if __name__ == "__main__":
    unittest.main()
