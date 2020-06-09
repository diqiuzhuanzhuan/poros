# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import tensorflow as tf
from unilmv2.config import Unilmv2Config
from poros_train.some_layer import (
    PositionEmbeddingLayer,
    EmbeddingLookupLayer,
    TransformerLayer,
    TokenTypeEmbeddingLayer,
    create_initializer
)
from poros_dataset import about_tensor
from poros_train.acitvation_function import get_activation


class Unilmv2Model(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(Unilmv2Model, self).__init__(*args, **kwargs)
        self.config = kwargs["config"]

    def call(self, inputs, **kwargs):
        pass


class InputEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, config: Unilmv2Config):
        super(InputEmbeddingLayer, self).__init__()
        if isinstance(config, Unilmv2Config):
            raise TypeError("config type muse be Unilmv2Config")
        self.config = config
        self.embedding_lookup_layer = EmbeddingLookupLayer(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range)

        self.position_embedding_layer = PositionEmbeddingLayer(
            position_size=config.max_position_embeddings,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range)

        self.token_type_embedding_layer = TokenTypeEmbeddingLayer(
            token_type_size=config.type_vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range)
        with tf.name_scope("LayerNorm"):
            self.normalization_layer = tf.keras.layers.LayerNormalization(epsilon=0.00001)
            self.normalization_layer.build(input_shape=[None, None, config.hidden_size])

    def call(self, inputs, **kwargs):
        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        token_type_ids = inputs["token_type_ids"]
        word_embeddings, _= self.embedding_lookup_layer(
            inputs=input_ids
        )
        position_embeddings, _ = self.position_embedding_layer(
            inputs=position_ids
        )
        token_type_embeddings, _ = self.token_type_embedding_layer(
            inputs=token_type_ids
        )
        input_embeddings = word_embeddings + position_embeddings + token_type_embeddings
        input_embeddings = self.normalization_layer(inputs=input_embeddings)

        return input_embeddings


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = about_tensor.get_shape(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


class MaskLmLayer(tf.keras.layers.Layer):

    def __init__(self, config: Unilmv2Config):
        if not isinstance(config, Unilmv2Config):
            raise TypeError("config type muse be Unilmv2Config")
        super(MaskLmLayer, self).__init__()
        self.config = config
        with tf.name_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.name_scope("transform"):
                with tf.name_scope("dense"):
                    self.layer_dense = tf.keras.layers.Dense(
                        units=self.config.hidden_size,
                        activation=get_activation(self.config.hidden_act),
                        kernel_initializer=create_initializer(self.config.initializer_range)
                    )
                    self.layer_dense.build(input_shape=[None, self.config.hidden_size])

                with tf.name_scope("LayerNorm"):
                    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=0.00001)
                    self.layer_norm.build(input_shape=[None, self.config.hidden_size])

            self.output_bias = tf.Variable(
                name="output_bias",
                initial_value=tf.zeros_initializer()(shape=[self.config.vocab_size]))

    def call(self, input_tensor, output_weights, positions, label_ids, label_weights):
        input_tensor = gather_indexes(input_tensor, positions)
        input_tensor = self.layer_dense(input_tensor)
        input_tensor = self.layer_norm(input_tensor)
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=self.config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        # per_example_loss tensor shape is `[batch_size * seq_length]`
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        # loss tensor shape is `[]`
        loss = numerator / denominator

        return (loss, per_example_loss, log_probs)


class PseudoMaskLmLayer(tf.keras.layers.Layer):

    def __init__(self, config: Unilmv2Config):
        if not isinstance(config, Unilmv2Config):
            raise TypeError("config type muse be Unilmv2Config")
        super(PseudoMaskLmLayer, self).__init__()
        self.config = config
        with tf.name_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.name_scope("transform"):
                with tf.name_scope("dense"):
                    self.layer_dense = tf.keras.layers.Dense(
                        units=self.config.hidden_size,
                        activation=get_activation(self.config.hidden_act),
                        kernel_initializer=create_initializer(self.config.initializer_range)
                    )
                    self.layer_dense.build(input_shape=[None, self.config.hidden_size])

                with tf.name_scope("LayerNorm"):
                    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=0.00001)
                    self.layer_norm.build(input_shape=[None, self.config.hidden_size])

            self.output_bias = tf.Variable(
                name="output_bias",
                initial_value=tf.zeros_initializer()(shape=[self.config.vocab_size]))

    def call(self, input_tensor, output_weights, positions, label_ids, label_weights):

        input_tensor = gather_indexes(input_tensor, positions)
        input_tensor = self.layer_dense(input_tensor)
        input_tensor = self.layer_norm(input_tensor)
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=self.config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        # per_example_loss tensor shape is `[batch_size * seq_length]`
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        # loss tensor shape is `[]`
        loss = numerator / denominator

        return (loss, per_example_loss, log_probs)


class NextSentenceLayer(tf.keras.layers.Layer):

    def __init__(self, config: Unilmv2Config):
        if not isinstance(config, Unilmv2Config):
            raise TypeError("config type muse be Unilmv2Config")
        super(NextSentenceLayer, self).__init__()
        self.config = config
        with tf.name_scope("cls/seq_relationship"):
            self.output_weights = tf.Variable(
                name="output_weights",
                initial_value=create_initializer(self.bert_config.initializer_range)(
                    shape=[2, self.bert_config.hidden_size])
            )

            self.output_bias = tf.Variable(
                name="output_bias",
                initial_value=tf.initializers.zeros()(shape=[2])
            )

    def call(self, input_tensor, labels):
        logits = tf.matmul(input_tensor, self.output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        # loss shape: [], per_example_loss shape [batch_size]
        # log_probs shape: [batch_size, 2]
        return (loss, per_example_loss, log_probs)
