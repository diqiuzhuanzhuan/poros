# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import unittest
from poros.bert import modeling
from poros.bert import run_pretraining
import tensorflow as tf
from poros.bert import tokenization
from poros.bert.run_classifier import SiameseProcessor


class TestRunPretraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        import logging
        tf.get_logger().setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.INFO)

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
        bert_config = modeling.BertConfig.from_json_file("../bert_model/test_data/bert_config.json")
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
        """
        features = {
            "input_ids": tf.keras.Input(shape=[max_seq_length], dtype=tf.int32, name="input_ids"),
            "input_mask": tf.keras.Input(shape=[max_seq_length], dtype=tf.int32, name="input_mask"),
            "segment_ids": tf.keras.Input(shape=[max_seq_length], dtype=tf.int32, name="segment_ids"),
            "masked_lm_positions": tf.keras.Input(shape=[max_predictions_per_seq], dtype=tf.int32, name="masked_lm_positions"),
            "masked_lm_ids": tf.keras.Input(shape=[max_predictions_per_seq], dtype=tf.int32, name="masked_lm_ids"),
            "masked_lm_weights": tf.keras.Input(shape=[max_predictions_per_seq], dtype=tf.float32, name="masked_lm_weights"),
            "next_sentence_labels": tf.keras.Input(shape=[1], dtype=tf.int32, name="next_sentence_labels")
        }
        """
        bert_pretrain_model = run_pretraining.BertPretrainModel(
            config=bert_config,
            is_training=True,
            init_checkpoint=None,
        )

        bert_pretrain_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001))
        d = tf.data.TFRecordDataset("./output")
        # Since we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        d = d.repeat()
        name_to_features = {
            "input_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.io.FixedLenFeature([1], tf.int64),
        }

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: run_pretraining._decode_record(record, name_to_features),
                batch_size=8,
                drop_remainder=True)).repeat()
        bert_pretrain_model.fit(d, epochs=500, steps_per_epoch=10, callbacks=[])

        output = bert_pretrain_model(features)
        print(output)

    def test_siamese_bert_model(self):
        bert_config = modeling.BertConfig.from_json_file("../bert_model/data/chinese_L-12_H-768_A-12/bert_config.json")
        tokenizer = tokenization.FullTokenizer("../bert_model/data/chinese_L-12_H-768_A-12/vocab.txt", do_lower_case=True)
        max_seq_length = 128
        batch_size = 8

        bert_siamese_model = run_pretraining.SiameseBertModel(
            config=bert_config,
            is_training=True,
            init_checkpoint="../bert_model/data/chinese_L-12_H-768_A-12/bert_model.ckpt"
        )

        bert_siamese_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
        siamese_processor = SiameseProcessor(tokenizer=tokenizer, max_seq_length=max_seq_length)
        lines = [("我想买保险", "我还想买保险", "0"),
                 ("我想买保险", "我不想买保险", "1"),
                 ("我想去北京玩耍", "我打算去北京玩耍", "0"),
                 ("我想去北京玩耍", "我不想去北京玩耍", "1"),
                 ("能不能借钱给我呢", "可以借钱给我吗", "0"),
                 ("我想借你钱", "我不想借你钱", "1")]

        examples = siamese_processor._create_examples(lines, set_type="dev")
        data = siamese_processor.convert_examples_to_features(examples)
        siamese_processor.write_features_into_tfrecord(filename="./siamese_data", features=data)
        d = siamese_processor.read_features_from_tfrecord(filename="./siamese_data", batch_size=6)
        d = d.repeat()

        bert_siamese_model.fit(d, epochs=5, steps_per_epoch=20)
        bert_siamese_model.evaluate(d, steps=3)
        print(bert_siamese_model.predict(d, steps=1))


if __name__ == "__main__":
    unittest.main()
