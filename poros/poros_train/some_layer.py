# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
from poros.poros_dataset import about_tensor


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def dot_product_attention(q, k, v, scale=True, bias=None, mask=None):
    """

    :param q:
    :param k:
    :param v:
    :param bias:
    :param dropout_rate:
    :param type:
    :return:
    """
    with tf.name_scope("dot_product_attention"):
        logits = tf.matmul(q, k, transpose_b=True)

        if bias:
            logits += bias
        if scale:
            logits = logits / tf.sqrt(tf.cast(about_tensor.get_shape(v)[-1], tf.float32))
        if mask:
            logits += mask * 1e-9
        weights = tf.nn.softmax(logits, name="attention_weights")

    return tf.matmul(weights, v)


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 name_scope="attention"):
        """

        :rtype: object
        """
        super(AttentionLayer, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.query_act = query_act
        self.key_act = key_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor
        self.wq = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
                                        activation=query_act,
                                        kernel_initializer=create_initializer(initializer_range=0.01))
        self.wk = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
                                        activation=key_act,
                                        kernel_initializer=create_initializer(initializer_range=0.01))
        self.wv = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
                                        activation=value_act,
                                        kernel_initializer=create_initializer(initializer_range=0.01))
        with tf.name_scope("query"):
            self.wq.build(input_shape=[None, size_per_head * num_attention_heads])
        with tf.name_scope("key"):
            self.wk.build(input_shape=[None, size_per_head * num_attention_heads])
        with tf.name_scope("value"):
            self.wv.build(input_shape=[None, size_per_head * num_attention_heads])

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, shape=[batch_size, -1, self.num_attention_heads, self.size_per_head])
        # shape is [B, N, F, H]
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, q, k, v,
             attention_mask=None,
             batch_size=None,
             from_seq_length=None,
             to_seq_length=None):
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head
        # q : [B, F, H]

        # convert to [B * F, N * H]

        from_shape = about_tensor.get_shape(q, expected_rank=[2, 3])
        to_shape = about_tensor.get_shape(k, expected_rank=[2, 3])

        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")

        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if (batch_size is None or from_seq_length is None or to_seq_length is None):
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")

        q = about_tensor.reshape_to_matrix(q)
        # convert to [B * T, N * H]
        k = about_tensor.reshape_to_matrix(k)
        v = about_tensor.reshape_to_matrix(v)

        # query_layer: [B, F, N*H]
        query_layer = self.wq(q)
        key_layer = self.wk(k)
        value_layer = self.wv(v)

        query_layer = self.transpose_for_scores(query_layer, batch_size)
        key_layer = self.transpose_for_scores(key_layer, batch_size)
        # attention_score's shape is [B, N, F, T]
        attention_score = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_score = attention_score / tf.math.sqrt(float(self.size_per_head))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_score += adder

        attention_score = tf.nn.softmax(attention_score)
        value_layer = self.transpose_for_scores(value_layer, batch_size)
        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_score, value_layer)
        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        # we need to concat all heads, now
        if self.do_return_2d_tensor:
            # `context_layer` = [B * F, N * H]
            context_layer = tf.reshape(context_layer,
                                       [-1,
                                        self.num_attention_heads * self.size_per_head])
        else:
            # `context_layer` = [B, F, N * H]
            context_layer = tf.reshape(context_layer,
                                       [batch_size, -1, self.num_attention_heads * self.size_per_head])

        return context_layer


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

