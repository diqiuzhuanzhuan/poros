# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import tensorflow as tf
import os
from poros_dataset import about_tensor
from poros.unilmv2 import PreTrainingDataMan
from poros.unilmv2.config import Unilmv2Config
from poros.unilmv2 import (
    Unilmv2Layer,
    MaskLmLayer,
    PseudoMaskLmLayer
)
from tensorflow.keras.callbacks import ReduceLROnPlateau


def pretrain():
    vocab_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "vocab.txt")
    ptdm = PreTrainingDataMan(vocab_file=vocab_file, max_seq_length=128, max_predictions_per_seq=1, random_seed=2334)
    input_file = "../bert/sample_text.txt"
    output_file = "./pretraining_data"
    ptdm.create_pretraining_data(input_file, output_file)
    dataset = ptdm.read_data_from_tfrecord(output_file, is_training=True, batch_size=8)
    dataset = dataset.repeat()
    json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "bert_config.json")
    unilmv2_config = Unilmv2Config.from_json_file(json_file)
    unilmv2_model = Unilmv2Model(config=unilmv2_config, is_training=True)
    reduce_lr = ReduceLROnPlateau(monitor='masked_lm_loss', factor=0.8,
                                  patience=10, min_lr=0)

    unilmv2_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=6e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-6))
    unilmv2_model.fit(dataset, epochs=2000, steps_per_epoch=15, callbacks=[reduce_lr])


class Unilmv2Model(tf.keras.Model):

    def __init__(self, config: Unilmv2Config, is_training=False, **kwargs):
        if not isinstance(config, Unilmv2Config):
            raise TypeError("config type muse be Unilmv2Config")
        super(Unilmv2Model, self).__init__(**kwargs)
        self.config = config
        self.unilmv2_layer = Unilmv2Layer(config=self.config, is_training=is_training)
        self.mask_lm_layer = MaskLmLayer(config=self.config)
        self.pseudo_mask_lm_layer = PseudoMaskLmLayer(config=self.config)
        self.masked_lm_accuracy_metric = tf.keras.metrics.Accuracy(name="masked_lm_accuracy")
        self.pseudo_masked_lm_accuracy_metric = tf.keras.metrics.Accuracy(name="pseudo_masked_lm_accuracy")
        self.masked_lm_loss = tf.keras.metrics.Mean(name="masked_lm_loss")
        self.pseudo_masked_lm_loss = tf.keras.metrics.Mean(name="pseudo_masked_lm_loss")

    def call(self, inputs):
        self.unilmv2_layer(inputs)
        masked_lm_input = self.unilmv2_layer.get_sequence_output()

        masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs = self.mask_lm_layer(
            masked_lm_input,
            self.unilmv2_layer.get_embedding_table(),
            inputs["masked_index"],
            inputs["masked_lm_ids"],
            inputs["masked_lm_weights"]
        )
        pseudo_lm_input = self.unilmv2_layer.get_sequence_output()
        pseudo_masked_lm_loss, pseudo_masked_lm_example_loss, pseudo_masked_lm_log_probs = self.pseudo_mask_lm_layer(
            pseudo_lm_input,
            self.unilmv2_layer.get_embedding_table(),
            inputs["pseudo_masked_index"],
            inputs["pseudo_masked_lm_ids"],
            inputs["masked_lm_weights"]
        )
        total_loss = masked_lm_loss + pseudo_masked_lm_loss
        self.add_loss(total_loss)

        masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)

        masked_lm_predictions = tf.reshape(masked_lm_predictions, shape=about_tensor.get_shape(inputs["masked_lm_ids"]))
        masked_lm_accuracy_metric = self.masked_lm_accuracy_metric(y_pred=masked_lm_predictions,
                                                                   y_true=inputs["masked_lm_ids"],
                                                                   sample_weight=inputs["masked_lm_weights"])
        pseudo_masked_lm_predictions = tf.argmax(pseudo_masked_lm_log_probs, axis=-1, output_type=tf.int32)
        pseudo_masked_lm_predictions = tf.reshape(pseudo_masked_lm_predictions, shape=about_tensor.get_shape(inputs["pseudo_masked_lm_ids"]))
        pseudo_masked_lm_accuracy_metric = self.pseudo_masked_lm_accuracy_metric(y_pred=pseudo_masked_lm_predictions,
                                                                   y_true=inputs["pseudo_masked_lm_ids"],
                                                                   sample_weight=inputs["masked_lm_weights"])

        masked_lm_loss_metric = self.masked_lm_loss(masked_lm_loss)
        pseudo_masked_lm_loss_metric = self.pseudo_masked_lm_loss(pseudo_masked_lm_loss)

        if not self.build:
            self.add_metric(masked_lm_accuracy_metric)
            self.add_metric(pseudo_masked_lm_accuracy_metric)
            self.add_metric(masked_lm_loss_metric)
            self.add_metric(pseudo_masked_lm_loss_metric)

        return total_loss


if __name__ == "__main__":
    pretrain()
