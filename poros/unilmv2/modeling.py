# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import tensorflow as tf
from poros.unilmv2 import Unilmv2Config
from poros.poros_train.some_layer import (
    PositionEmbeddingLayer,
    TokenTypeEmbeddingLayer,
    EmbeddingLookupLayer,
    create_initializer,
    TransformerLayer
)
from poros.poros_dataset import about_tensor
from poros.poros_train import acitvation_function
import copy


class Unilmv2Layer(tf.keras.layers.Layer):

    def __init__(self, config: Unilmv2Config, is_training=False, **kwargs):
        if not isinstance(config, Unilmv2Config):
            raise TypeError("config type muse be Unilmv2Config")
        super(Unilmv2Layer, self).__init__(**kwargs)
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        self.config = config
        with tf.name_scope(name="unilmv2"):
            with tf.name_scope(name="embeddings"):
                self.input_embedding_layer = InputEmbeddingLayer(self.config)
            with tf.name_scope(name="encode"):
                self.transformer_layer = TransformerLayer(
                    hidden_size=self.config.hidden_size,
                    num_hidden_layers=self.config.num_hidden_layers,
                    num_attention_heads=self.config.num_attention_heads,
                    intermediate_size=self.config.intermediate_size,
                    intermediate_act_fn=self.config.hidden_act,
                    hidden_dropout_prob=self.config.hidden_dropout_prob,
                    attention_probs_dropout_prob=self.config.hidden_dropout_prob,
                    initializer_range=self.config.initializer_range,
                    do_return_all_layers=True
                )
            with tf.name_scope("pooler/dense"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                self.pooler_layer = tf.keras.layers.Dense(
                    self.config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(
                        self.config.initializer_range))
                self.pooler_layer.build(input_shape=[None, config.hidden_size])

    def call(self, inputs, training=False, **kwargs):
        input_ids = inputs["input_ids"]
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
        else:
            attention_mask = None
        output_tokens_positions = inputs["output_tokens_positions"]
        segment_ids = inputs["segment_ids"]

        self.embedding_output, self.embedding_table = self.input_embedding_layer({
            "input_ids": input_ids,
            "position_ids": output_tokens_positions,
            "segment_ids": segment_ids
        })
        self.all_encoder_layers = self.transformer_layer(inputs=self.embedding_output, attention_mask=attention_mask, training=training)
        #self.all_encoder_layers = transformer_model(self.embedding_output, attention_mask=None, do_return_all_layers=True)
        self.sequence_output = self.all_encoder_layers[-1]
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        #self.pooled_output = self.pooler_layer(first_token_tensor)
        #return self.pooled_output
        return self.sequence_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the output of the embedding layer, after summing the word
          embeddings with the positional embeddings and the token type embeddings,
          then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


class InputEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, config: Unilmv2Config):
        super(InputEmbeddingLayer, self).__init__()
        if not isinstance(config, Unilmv2Config):
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
            token_type_size=2,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range
        )

        with tf.name_scope("LayerNorm"):
            self.normalization_layer = tf.keras.layers.LayerNormalization(epsilon=0.00001)
            self.normalization_layer.build(input_shape=[None, None, config.hidden_size])

    def call(self, inputs, **kwargs):
        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        segment_ids = inputs["segment_ids"]
        word_embeddings, embedding_table = self.embedding_lookup_layer(
            input_ids
        )
        position_embeddings, _ = self.position_embedding_layer(
            position_ids
        )
        segment_embeddings, _ = self.token_type_embedding_layer(
            segment_ids
        )
        input_embeddings = word_embeddings + position_embeddings + segment_embeddings
        input_embeddings = self.normalization_layer(inputs=input_embeddings)

        return input_embeddings, embedding_table


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = about_tensor.get_shape(sequence_tensor, expected_rank=3)
    positions_shape = about_tensor.get_shape(positions, expected_rank=2)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]
    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * positions_shape[1], [-1, 1])
    flat_offsets = tf.cast(flat_offsets, dtype=tf.int64)
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
        with tf.name_scope("cls/predictions/mask_lm"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.name_scope("transform"):
                with tf.name_scope("dense"):
                    self.layer_dense = tf.keras.layers.Dense(
                        units=self.config.hidden_size,
                        activation=acitvation_function.get_activation(self.config.hidden_act),
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
        with tf.name_scope("cls/predictions/pseudo_lm"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.name_scope("transform"):
                with tf.name_scope("dense"):
                    self.layer_dense = tf.keras.layers.Dense(
                        units=self.config.hidden_size,
                        activation=acitvation_function.get_activation(self.config.hidden_act),
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


if __name__ == "__main__":
    tf.debugging.enable_check_numerics()
    ul = Unilmv2Layer(config=Unilmv2Config(vocab_size=100))
    input_ids = tf.random.uniform(shape=[3, 20], dtype=tf.int32, maxval=100)
    input_mask = tf.ones(shape=[3, 20])
    pseudo_index = tf.constant(
        value=[[3, 7, 9, 10], [4, 9, 8, 15], [5, 11, 17, 18]]
    )
    pseudo_len = tf.constant(
        value=[[2, 1, 1, 0], [1, 1, 1, 1], [3, 1, 0, 0]]
    )

