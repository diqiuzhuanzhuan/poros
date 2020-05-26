# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import tensorflow as tf
from poros_dataset import about_tensor


class EmbeddingLookupLayer(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size=128, initializer_range=0.02, name="word_embeddings"):
        super(EmbeddingLookupLayer, self).__init__()
        truncated_normal = tf.initializers.TruncatedNormal(stddev=initializer_range)
        self.embedding_table = \
            tf.Variable(name=name,
                        initial_value=truncated_normal(shape=[vocab_size, embedding_size]))
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def call(self, input_ids, use_one_hot_embeddings=False):
        """Looks up words embeddings for id tensor.

        Args:
            input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
            use_one_hot_embeddings: boolean, use one hot form

        Returns:
            float Tensor of shape [batch_size, seq_length, embedding_size].

        """
        # If the input is a 2D tensor of shape [batch_size, seq_length], we
        # reshape to [batch_size, seq_length, 1].
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])

        flat_input_ids = tf.reshape(input_ids, [-1])
        if use_one_hot_embeddings:
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=self.vocab_size)
            output = tf.matmul(one_hot_input_ids, self.embedding_table)
        else:
            output = tf.gather(self.embedding_table, flat_input_ids)

        input_shape = about_tensor.get_shape(input_ids)

        output = tf.reshape(output,
                            input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
        return output, self.embedding_table


class PositionEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, position_size, embedding_size=128, initializer_range=0.02, name="position_embeddings"):
        super(PositionEmbeddingLayer, self).__init__()
        truncated_normal = tf.initializers.TruncatedNormal(stddev=initializer_range)
        self.embedding_table = \
            tf.Variable(name=name, initial_value=truncated_normal(shape=[position_size, embedding_size]))
        self.position_size = position_size
        self.embedding_size = embedding_size

    def call(self, input_ids, use_one_hot_embeddings=False):
        """Looks up position embeddings for id tensor.

        Args:
            input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
            use_one_hot_embeddings: boolean, use one hot form

        Returns:
            float Tensor of shape [batch_size, seq_length, embedding_size].

        """
        # If the input is a 2D tensor of shape [batch_size, seq_length], we
        # reshape to [batch_size, seq_length, 1].
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])
        flat_input_ids = tf.reshape(input_ids, [-1])
        if use_one_hot_embeddings:
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=self.position_size)
            output = tf.matmul(one_hot_input_ids, self.embedding_table)
        else:
            output = tf.gather(self.embedding_table, flat_input_ids)

        input_shape = about_tensor.get_shape(input_ids)
        output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
        return output, self.embedding_table
