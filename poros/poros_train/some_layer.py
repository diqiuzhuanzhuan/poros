# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
from poros.poros_dataset import about_tensor
from poros.poros_train import acitvation_function
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import InputSpec
import numpy as np


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

    output = tf.nn.dropout(input_tensor, dropout_prob)
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


def create_initializer(initializer_range=0.02, name="truncated_normal"):
    """Creates a `truncated_normal_initializer` with the given range."""
    if name == "truncated_normal":
        return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
    elif name == "random_normal":
        return tf.keras.initializers.RandomNormal(stddev=initializer_range)
    else:
        raise ValueError("no initializer named :{}".format(name))


def create_rpr_matrix(k, from_length, to_length):
    """
    we want a matrix like below:
    [
        [ 0,  1, 2, 3, ..., k, k, k],
        [-1,  0, 1, 2, ..., k, k, k],
        [-2, -1, 0, 1, ..., k, k, k],
        ...
        [-k, -k, -k,  ...,  0,  1,  2],
        [-k, -k, -k,  ...,  -1, 0,  1],
        [-k, -k, -k,  ...,  -2, -1, 0],
    ]

    :param k:
    :param from_length:
    :param to_length:
    :return:
    """

    def __clip(x, k):
        return max(-k, min(k, x))

    matrix = np.ones(shape=[from_length, to_length], dtype=np.int)
    for i in range(from_length):
        for j in range(to_length):
            matrix[i, j] = __clip(j - i, k)
    matrix += k
    return matrix


class AttentionLayerWithRPR(tf.keras.layers.Layer):

    def __init__(self,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 clip_k=5,
                 name_scope="attention"):
        """

        :rtype: object
        """
        super(AttentionLayerWithRPR, self).__init__(name=name_scope)
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.query_act = query_act
        self.key_act = key_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor
        self.clip_k = clip_k
        self.wq = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
                                        activation=query_act,
                                        kernel_initializer=create_initializer(initializer_range=0.01))
        self.wk = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
                                        activation=key_act,
                                        kernel_initializer=create_initializer(initializer_range=0.01))
        self.wv = tf.keras.layers.Dense(self.num_attention_heads * self.size_per_head,
                                        activation=value_act,
                                        kernel_initializer=create_initializer(initializer_range=0.01))
        self.krpr = tf.Variable(name="krpr",
                                initial_value=tf.initializers.TruncatedNormal()(shape=[2*clip_k+1, self.size_per_head]))
        self.vrpr = tf.Variable(name="vrpr",
                                initial_value=tf.initializers.TruncatedNormal()(shape=[2*clip_k+1, self.size_per_head]))

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
             to_seq_length=None,
             rpr_matrix=None
             ):
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
        # query_layer = [B, N, F, H]
        query_layer = self.transpose_for_scores(query_layer, batch_size)
        key_layer = self.transpose_for_scores(key_layer, batch_size)
        # attention_score's shape is [B, N, F, T]
        attention_score = tf.matmul(query_layer, key_layer, transpose_b=True)
        if rpr_matrix is not None:
            alpha = tf.gather(self.krpr, rpr_matrix)
            origin_shape = about_tensor.get_shape(query_layer)
            query_layer = tf.reshape(query_layer, shape=[batch_size, self.num_attention_heads, from_seq_length, 1, self.size_per_head])
            alpha_score = tf.matmul(query_layer, alpha, transpose_b=True)
            alpha_score = tf.reshape(alpha_score, shape=[batch_size, self.num_attention_heads, from_seq_length, to_seq_length])
            attention_score = alpha_score + attention_score
            query_layer = tf.reshape(query_layer, shape=origin_shape)

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
        # `value_layer` = [B, N, T, H]
        value_layer = self.transpose_for_scores(value_layer, batch_size)

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_score, value_layer)
        if rpr_matrix is not None:
            # `alpha` = [F, T, H]
            alpha = tf.gather(self.krpr, rpr_matrix)
            origin_shape = about_tensor.get_shape(attention_score)
            attention_score = tf.reshape(attention_score, shape=[batch_size, self.num_attention_heads, from_seq_length, 1, to_seq_length])
            alpha_layer = tf.reshape(tf.matmul(attention_score, alpha), shape=[batch_size, self.num_attention_heads, to_seq_length, self.size_per_head])
            context_layer = alpha_layer + context_layer
            attention_score = tf.reshape(attention_score, origin_shape)
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
        super(AttentionLayer, self).__init__(name=name_scope)
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
                                       [batch_size * from_seq_length,
                                        self.num_attention_heads * self.size_per_head])
        else:
            # `context_layer` = [B, F, N * H]
            context_layer = tf.reshape(context_layer,
                                       [batch_size, from_seq_length, self.num_attention_heads * self.size_per_head])

        return context_layer


class TransformerLayer(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 intermediate_act_fn=acitvation_function.gelu,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 do_return_all_layers=False,
                 ):
        """

        :param hidden_size:
        :param num_hidden_layers:
        :param num_attention_heads:
        :param intermediate_size:
        :param intermediate_act_fn:
        :param hidden_dropout_prob:
        :param attention_probs_dropout_prob:
        :param initializer_range:
        :param do_return_all_layers:
        """
        super(TransformerLayer, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.do_return_all_layers = do_return_all_layers
        self.attention_layers = []
        self.attention_outputs = []
        self.attention_outputs_layer_norm = []
        self.attention_outputs_rezero_layers = []
        self.intermediate_outputs = []
        self.dense_output_rezero_layers = []
        self.size_per_head = int(self.hidden_size / self.num_attention_heads)
        self.outputs = []
        self.outputs_layer_norm = []
        for layer_idx in range(self.num_hidden_layers):
            with tf.name_scope("layer_%d" % layer_idx):
                with tf.name_scope("attention"):
                    with tf.name_scope("self"):
                        layer = AttentionLayer(
                            num_attention_heads=self.num_attention_heads,
                            size_per_head=self.size_per_head,
                            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                            initializer_range=self.initializer_range,
                        )
                        self.attention_layers.append(layer)
                    with tf.name_scope("output/dense"):
                        layer = tf.keras.layers.Dense(
                            hidden_size,
                            kernel_initializer=create_initializer(self.initializer_range))
                        layer.build(input_shape=[None, self.hidden_size])
                        self.attention_outputs.append(layer)
                    with tf.name_scope("output/norm"):
                        layer = tf.keras.layers.LayerNormalization()
                        self.attention_outputs_layer_norm.append(layer)
                    with tf.name_scope("output/RezeroLayer"):
                        layer = RezeroLayer()
                        self.attention_outputs_rezero_layers.append(layer)

                with tf.name_scope("intermediate/dense"):
                    layer = tf.keras.layers.Dense(
                        self.intermediate_size,
                        activation=acitvation_function.get_activation(self.intermediate_act_fn),
                        kernel_initializer=create_initializer(self.initializer_range))
                    layer.build(input_shape=[None, self.hidden_size])
                    self.intermediate_outputs.append(layer)

                with tf.name_scope("output"):
                    with tf.name_scope("dense"):
                        layer = tf.keras.layers.Dense(
                            hidden_size,
                            kernel_initializer=create_initializer(self.initializer_range))
                        layer.build(input_shape=[None, self.intermediate_size])
                        self.outputs.append(layer)

                    with tf.name_scope("LayerNorm"):
                        layer = tf.keras.layers.LayerNormalization(epsilon=0.00001)
                        layer.build(input_shape=[None, None, self.hidden_size])
                        self.outputs_layer_norm.append(layer)

                    with tf.name_scope("rezero"):
                        layer = RezeroLayer()
                        self.dense_output_rezero_layers.append(layer)

    def call(self, inputs, attention_mask, training=False):
        input_tensor = inputs
        input_shape = about_tensor.get_shape(input_tensor, expected_rank=3)
        input_width = input_shape[2]

        if input_width % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (input_width, self.num_attention_heads))

        size_per_head = int(input_width / self.num_attention_heads)
        tf.debugging.assert_equal(size_per_head, self.size_per_head)

        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        # prev_output = reshape_to_matrix(input_tensor)
        prev_output = input_tensor

        all_layer_outputs = []

        for (attention_layer, attention_output_layer, intermediate_output, output,
             attention_outputs_rezero_layer, dense_output_rezero_layer,attention_outputs_layer_norm,
             outputs_layer_norm) \
                in zip(self.attention_layers,
                       self.attention_outputs,
                       self.intermediate_outputs,
                       self.outputs,
                       self.attention_outputs_rezero_layers,
                       self.dense_output_rezero_layers,
                       self.attention_outputs_layer_norm,
                       self.outputs_layer_norm
                       ):
            layer_input = prev_output
            attention_heads = []
            attention_head = attention_layer(layer_input, layer_input, layer_input, attention_mask)
            attention_heads.append(attention_head)
            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                # In the case where we have other sequences, we just concatenate
                # them to the self-attention head before the projection.
                attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
            attention_output = attention_output_layer(attention_output)
            if training:
                attention_output = dropout(attention_output, self.hidden_dropout_prob)

            attention_output = attention_outputs_layer_norm(layer_input+attention_output)
            if training:
                attention_output = dropout(attention_output, self.hidden_dropout_prob)

            # The activation is only applied to the "intermediate" hidden layer.
            intermediate_output = intermediate_output(attention_output)

            # Down-project back to `hidden_size` then add the residual.
            layer_output = output(intermediate_output)
            if training:
                layer_output = dropout(layer_output, self.hidden_dropout_prob)
            layer_output = outputs_layer_norm(layer_output + attention_output)
            if training:
                layer_output = dropout(layer_output, self.hidden_dropout_prob)
            prev_output = layer_output
            all_layer_outputs.append(layer_output)

        if self.do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = about_tensor.reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = about_tensor.reshape_from_matrix(prev_output, input_shape)
            return final_output


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


class TokenTypeEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, token_type_size, embedding_size=128, initializer_range=0.02, name="token_type_embeddings"):
        super(TokenTypeEmbeddingLayer, self).__init__()
        truncated_normal = tf.initializers.TruncatedNormal(stddev=initializer_range)
        self.embedding_table = tf.Variable(name=name, initial_value=truncated_normal(shape=[token_type_size, embedding_size]))
        self.token_type_size = token_type_size
        self.embedding_size = embedding_size

    def call(self, input_ids, use_one_hot_embeddings=False):
        """Looks up position embeddings for id tensor.

               Args:
                   input_ids: int32 Tensor of shape [batch_size, seq_length] containing type of tokens
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


class RezeroLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=True):
        super(RezeroLayer, self).__init__(trainable=trainable)

    def build(self, input_shape):
        ndims = len(input_shape)
        if ndims is None:
            raise ValueError('Input shape %s has undefined rank.' % input_shape)
        if ndims < 2:
            raise ValueError('Input shape %s must be longer than 1.' % input_shape)

        tensor_shape.TensorShape(input_shape)
        shape = [tensor_shape.dimension_value(input_shape[-2]), tensor_shape.dimension_value(input_shape[-1])]
        self.alpha = self.add_weight(name="alpha",
                        shape=[1],
                        initializer=tf.initializers.Zeros,
                        trainable=True)
        self.built = True

    def call(self, inputs):
        outputs = tf.multiply(self.alpha, inputs)
        return outputs
    

class MMoELayer(tf.keras.layers.Layer):
    """
    参数为专家个数

    """

    def __init__(self,
                 units,
                 num_experts,
                 num_tasks,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """
         Method for instantiating MMoE layer.
        :param units: Number of hidden units
        :param num_experts: Number of experts
        :param num_tasks: Number of tasks
        :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights
        :param expert_activation: Activation function of the expert weights
        :param gate_activation: Activation function of the gate weights
        :param expert_bias_initializer: Initializer for the expert bias
        :param gate_bias_initializer: Initializer for the gate bias
        :param expert_bias_regularizer: Regularizer for the expert bias
        :param gate_bias_regularizer: Regularizer for the gate bias
        :param expert_bias_constraint: Constraint for the expert bias
        :param gate_bias_constraint: Constraint for the gate bias
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class
        """
        # Hidden nodes parameter
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        # Activation parameter
        self.expert_activation = activations.get(expert_activation)
        self.gate_activation = activations.get(gate_activation)

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Keras parameter
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        super(MMoELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Method for creating the layer weights.
        :param input_shape: Keras tensor (future input to layer)
                            or list/tuple of Keras tensors to reference
                            for weight shape computations
        """
        assert input_shape is not None and len(input_shape) >= 2

        input_dimension = input_shape[-1]

        # Initialize expert weights (number of input features * number of units per expert * number of experts)
        self.expert_kernels = self.add_weight(
            name='expert_kernel',
            shape=(input_dimension, self.units, self.num_experts),
            initializer=self.expert_kernel_initializer,
            regularizer=self.expert_kernel_regularizer,
            constraint=self.expert_kernel_constraint,
        )

        # Initialize expert bias (number of units per expert * number of experts)
        if self.use_expert_bias:
            self.expert_bias = self.add_weight(
                name='expert_bias',
                shape=(self.units, self.num_experts),
                initializer=self.expert_bias_initializer,
                regularizer=self.expert_bias_regularizer,
                constraint=self.expert_bias_constraint,
            )

        # Initialize gate weights (number of input features * number of experts * number of tasks)
        self.gate_kernels = [self.add_weight(
            name='gate_kernel_task_{}'.format(i),
            shape=(input_dimension, self.num_experts),
            initializer=self.gate_kernel_initializer,
            regularizer=self.gate_kernel_regularizer,
            constraint=self.gate_kernel_constraint
        ) for i in range(self.num_tasks)]

        # Initialize gate bias (number of experts * number of tasks)
        if self.use_gate_bias:
            self.gate_bias = [self.add_weight(
                name='gate_bias_task_{}'.format(i),
                shape=(self.num_experts,),
                initializer=self.gate_bias_initializer,
                regularizer=self.gate_bias_regularizer,
                constraint=self.gate_bias_constraint
            ) for i in range(self.num_tasks)]

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dimension})

        super(MMoELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Method for the forward function of the layer.
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        gate_outputs = []
        final_outputs = []

        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
        expert_outputs = tf.tensordot(a=inputs, b=self.expert_kernels, axes=1)
        # Add the bias term to the expert weights if necessary
        if self.use_expert_bias:
            expert_outputs = tf.keras.backend.bias_add(x=expert_outputs, bias=self.expert_bias)
        expert_outputs = self.expert_activation(expert_outputs)

        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = tf.keras.backend.dot(x=inputs, y=gate_kernel)
            # Add the bias term to the gate weights if necessary
            if self.use_gate_bias:
                gate_output = tf.keras.backend.bias_add(x=gate_output, bias=self.gate_bias[index])
            gate_output = self.gate_activation(gate_output)
            gate_outputs.append(gate_output)

        # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
        for gate_output in gate_outputs:
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * tf.keras.backend.repeat_elements(expanded_gate_output, self.units, axis=1)
            final_outputs.append(tf.keras.backend.sum(weighted_expert_output, axis=2))

        return final_outputs

    def compute_output_shape(self, input_shape):
        """
        Method for computing the output shape of the MMoE layer.
        :param input_shape: Shape tuple (tuple of integers)
        :return: List of input shape tuple where the size of the list is equal to the number of tasks
        """
        assert input_shape is not None and len(input_shape) >= 2

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)

        return [output_shape for _ in range(self.num_tasks)]

    def get_config(self):
        """
        Method for returning the configuration of the MMoE layer.
        :return: Config dictionary
        """

        config = {
            'units': self.units,
            'num_experts': self.num_experts,
            'num_tasks': self.num_tasks,
            'use_expert_bias': self.use_expert_bias,
            'use_gate_bias': self.use_gate_bias,
            'expert_activation': activations.serialize(self.expert_activation),
            'gate_activation': activations.serialize(self.gate_activation),
            'expert_bias_initializer': initializers.serialize(self.expert_bias_initializer),
            'gate_bias_initializer': initializers.serialize(self.gate_bias_initializer),
            'expert_bias_regularizer': regularizers.serialize(self.expert_bias_regularizer),
            'gate_bias_regularizer': regularizers.serialize(self.gate_bias_regularizer),
            'expert_bias_constraint': constraints.serialize(self.expert_bias_constraint),
            'gate_bias_constraint': constraints.serialize(self.gate_bias_constraint),
            'expert_kernel_initializer': initializers.serialize(self.expert_kernel_initializer),
            'gate_kernel_initializer': initializers.serialize(self.gate_kernel_initializer),
            'expert_kernel_regularizer': regularizers.serialize(self.expert_kernel_regularizer),
            'gate_kernel_regularizer': regularizers.serialize(self.gate_kernel_regularizer),
            'expert_kernel_constraint': constraints.serialize(self.expert_kernel_constraint),
            'gate_kernel_constraint': constraints.serialize(self.gate_kernel_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(MMoELayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

