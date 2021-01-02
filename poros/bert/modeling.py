# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at/dense
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import re
import numpy as np
import six
import tensorflow as tf
from tensorflow import *


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertLayer(tf.keras.layers.Layer):
    """BERT model ("Bidirectional Encoder Representations from Transformers").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
      num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
      input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.Variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    config = None

    def __init__(self,
                 config,
                 is_training):
        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.
          is_training: bool. true for training model, false for eval model. Controls
            whether dropout will be applied.
          input_ids: int32 Tensor of shape [batch_size, seq_length].
          input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
          token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
          use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings.
          scope: (optional) variable scope. Defaults to "bert".

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """

        super(BertLayer, self).__init__()
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        self.config = config
        with tf.name_scope(name="bert"):
            with tf.name_scope("embeddings"):
                self.embedding_layer = EmbeddingLookupLayer(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    self.config.initializer_range,
                    "word_embeddings"
                )
                self.embedding_postprocesser_layer = EmbeddingPostprocessorLayer(
                    use_token_type=True,
                    embedding_size=self.config.hidden_size,
                    max_position_embeddings=self.config.max_position_embeddings,
                    initializer_range=self.config.initializer_range,
                    token_type_vocab_size=self.config.type_vocab_size,
                )
            with tf.name_scope(name="encoder"):
                #from poros_train.some_layer import TransformerLayer
                self.transformer_layer = TransformerLayer(
                    hidden_size=self.config.hidden_size,
                    num_hidden_layers=self.config.num_hidden_layers,
                    num_attention_heads=self.config.num_attention_heads,
                    intermediate_size=self.config.intermediate_size,
                    intermediate_act_fn=get_activation(self.config.hidden_act),
                    hidden_dropout_prob=self.config.hidden_dropout_prob,
                    attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
                    initializer_range=self.config.initializer_range,
                    do_return_all_layers=True)

            with tf.name_scope("pooler/dense"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                self.pooler_layer = tf.keras.layers.Dense(
                    self.config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(
                        self.config.initializer_range))
                self.pooler_layer.build(input_shape=[None, config.hidden_size])

    def call(self,
             input_ids,
             input_mask=None,
             token_type_ids=None,
             scope="bert",
             use_one_hot_embeddings=False,
             training=False):
        """
        Args:
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            scope: variable scope name, defaults to `bert`
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
                embeddings or tf.embedding_lookup() for the word embeddings.
        """

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.name_scope(scope):
            with tf.name_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                self.embedding_output, self.embedding_table = \
                    self.embedding_layer(input_ids, use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = self.embedding_postprocesser_layer(
                    self.embedding_output,
                    token_type_ids,
                    self.config.hidden_dropout_prob,
                    training=training
                )
            with tf.name_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)

                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].

                self.all_encoder_layers = self.transformer_layer(self.embedding_output, attention_mask, training=training)

            self.sequence_output = self.all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.name_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = self.pooler_layer(first_token_tensor)
        return self.pooled_output

    def get_pooled_output(self):
        return self.pooled_output

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


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string: String name of the activation function.

    Returns:
      A Python function corresponding to the activation function. If
      `activation_string` is None, empty, or "linear", this will return None.
      If `activation_string` is not a string, it will return `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)
    tf.get_logger().info("variable num is {}".format(len(init_vars)))
    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


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


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.keras.layers.LayerNormalization(epsilon=0.00001, name=name)(input_tensor)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


class EmbeddingLookupLayer(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size=128, initializer_range=0.02, name="word_embeddings"):
        super(EmbeddingLookupLayer, self).__init__()
        truncated_normal = tf.initializers.TruncatedNormal(stddev=initializer_range)
        self.embedding_table = \
            tf.Variable(name=name,
                        initial_value=truncated_normal(shape=[vocab_size, embedding_size]))
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def call(self, input_ids, use_one_hot_embeddings):
        """Looks up words embeddings for id tensor.

        Args:
            input_ids: int32 Tensor of shape [batch_size, seq_length] containing word

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

        input_shape = get_shape_list(input_ids)

        output = tf.reshape(output,
                            input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
        return output, self.embedding_table



class EmbeddingPostprocessorLayer(tf.keras.layers.Layer):

    def __init__(self, use_token_type=False,
                 use_position_embeddings=True,
                 token_type_embedding_name="token_type_embeddings",
                 position_embedding_name="position_embeddings",
                 initializer_range=0.02,
                 token_type_vocab_size=16,
                 embedding_size=768,
                 max_position_embeddings=512):
        super(EmbeddingPostprocessorLayer, self).__init__()
        self.use_token_type = use_token_type
        self.use_position_embeddings = use_position_embeddings
        self.token_type_vocab_size = token_type_vocab_size
        self.embedding_size = embedding_size
        self.max_position_embeddings=512
        if self.use_token_type:
            self.token_type_table = tf.Variable(
                name=token_type_embedding_name,
                initial_value=create_initializer(initializer_range)(
                    shape=[self.token_type_vocab_size, self.embedding_size])
            )

        if use_position_embeddings:
            self.full_position_embeddings = tf.Variable(
                name=position_embedding_name,
                initial_value=create_initializer(initializer_range)(
                    shape=[max_position_embeddings, self.embedding_size])
            )
        with tf.name_scope("LayerNorm"):
            self.layer_normalization = tf.keras.layers.LayerNormalization(epsilon=0.00001)
            self.layer_normalization.build(input_shape=[None, None, self.embedding_size])

    def call(self, input_tensor, token_type_ids, dropout_prob=0.1, training=False):
        input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]
        output = input_tensor
        tf.debugging.assert_equal(
            tf.constant(value=width, dtype=tf.int32), tf.constant(value=self.embedding_size, dtype=tf.int32),
            message="the dimension of input tensor is not equal to the embedding size, "
                    "input_tensor is {}, embedding size is {}".format(width, self.embedding_size)
        )

        if self.use_token_type:
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])
            one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
            token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_table)
            token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])
            output += token_type_embeddings

        if self.use_position_embeddings:
            assert_op = tf.debugging.assert_less_equal(seq_length, self.max_position_embeddings)
            with tf.control_dependencies([assert_op]):
                position_embeddings = tf.slice(self.full_position_embeddings, [0, 0],
                                           [seq_length, -1])
                num_dims = len(output.shape.as_list())
                # Only the last two dimensions are relevant (`seq_length` and `width`), so
                # we broadcast among the first dimensions, which is typically just
                # the batch size.
                position_broadcast_shape = []
                for _ in output.shape.as_list()[0:-2]:
                    position_broadcast_shape.append(1)

                """
                for _ in range(num_dims - 2):
                    position_broadcast_shape.append(1)
                """
                position_broadcast_shape.extend([seq_length, width])
                position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
                output += position_embeddings

        output = self.layer_normalization(output)
        if training:
            output = dropout(output, dropout_prob)

        return output


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """Looks up words embeddings for id tensor.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
        ids.
      vocab_size: int. Size of the embedding vocabulary.
      embedding_size: int. Width of the word embeddings.
      initializer_range: float. Embedding initialization range.
      word_embedding_name: string. Name of the embedding table.
      use_one_hot_embeddings: bool. If True, use one-hot method for word
        embeddings. If False, use `tf.gather()`.

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.Variable(
        name=word_embedding_name,
        initial_value=tf.initializers.TruncatedNormal(stddev=0.01)(shape=[vocab_size, embedding_size]))

    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table, flat_input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    """Performs various post-processing on a word embedding tensor.

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length,
        embedding_size].
      use_token_type: bool. Whether to add embeddings for `token_type_ids`.
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        Must be specified if `use_token_type` is True.
      token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
      token_type_embedding_name: string. The name of the embedding table variable
        for token type ids.
      use_position_embeddings: bool. Whether to add position embeddings for the
        position of each token in the sequence.
      position_embedding_name: string. The name of the embedding table variable
        for positional embeddings.
      initializer_range: float. Range of the weight initialization.
      max_position_embeddings: int. Maximum sequence length that might ever be
        used with this model. This can be longer than the sequence length of
        input_tensor, but cannot be shorter.
      dropout_prob: float. Dropout probability applied to the final output tensor.

    Returns:
      float tensor with same shape as `input_tensor`.

    Raises:
      ValueError: One of the tensor shapes or input values is invalid.
    """
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.Variable(
            name=token_type_embedding_name,
            initial_value=create_initializer(initializer_range)(shape=[token_type_vocab_size, width]))
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        output += token_type_embeddings

    if use_position_embeddings:
        assert_op = tf.debugging.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.Variable(
                name=position_embedding_name,
                initial_value=create_initializer(initializer_range)(shape=[max_position_embeddings, width]))
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings,
                                             position_broadcast_shape)
            output += position_embeddings
    output = layer_norm_and_dropout(output, dropout_prob)
    return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.
    if the position is need to be masked, set 1 to it

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


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
        #self.attention_probs_dropout_prob = attention_probs_dropout_prob
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

        from_shape = get_shape_list(q, expected_rank=[2, 3])
        to_shape = get_shape_list(k, expected_rank=[2, 3])

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

        q = reshape_to_matrix(q)
        # convert to [B * T, N * H]
        k = reshape_to_matrix(k)
        v = reshape_to_matrix(v)

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


class TransformerLayer(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 intermediate_act_fn=gelu,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 do_return_all_layers=False):
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
        self.intermediate_outputs = []
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
                    with tf.name_scope("output/LayerNorm"):
                        layer = tf.keras.layers.LayerNormalization(epsilon=0.00001)
                        layer.build(input_shape=[None, None, self.hidden_size])
                        self.attention_outputs_layer_norm.append(layer)

                with tf.name_scope("intermediate/dense"):
                    layer = tf.keras.layers.Dense(
                        self.intermediate_size,
                        activation=self.intermediate_act_fn,
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

        #self.attention_layers = tf.stack(self.attention_layers)
        #self.attention_outputs = tf.stack(self.attention_outputs)
        #self.attention_outputs_layer_norm = tf.stack(self.attention_outputs_layer_norm)
        #self.intermediate_outputs = tf.stack(self.intermediate_outputs)
        #self.outputs = tf.stack(self.outputs)
        #self.outputs_layer_norm = tf.stack(self.outputs_layer_norm)

    def call(self, input_tensor, attention_mask, training=False):
        #input_tensor = features["input_tensor"]
        #attention_mask = features["attention_mask"]
        input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
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

        for (attention_layer,
             attention_output_layer,
             attention_output_layer_norm,
             intermediate_output,
             output,
             output_layer_norm) in zip(self.attention_layers,
                                   self.attention_outputs,
                                   self.attention_outputs_layer_norm,
                                   self.intermediate_outputs,
                                   self.outputs,
                                   self.outputs_layer_norm):
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
            attention_output = attention_output_layer_norm(attention_output + layer_input)
            if training:
                attention_output = dropout(attention_output, self.hidden_dropout_prob)

            # The activation is only applied to the "intermediate" hidden layer.
            intermediate_output = intermediate_output(attention_output)

            # Down-project back to `hidden_size` then add the residual.
            layer_output = output(intermediate_output)
            if training:
                layer_output = dropout(layer_output, self.hidden_dropout_prob)
            layer_output = output_layer_norm(layer_output + attention_output)
            if training:
                layer_output = dropout(layer_output, self.hidden_dropout_prob)
            prev_output = layer_output
            all_layer_outputs.append(layer_output)

        if self.do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = reshape_from_matrix(prev_output, input_shape)
            return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None and not tf.executing_eagerly():
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None and not tf.executing_eagerly():
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            "For the tensor `%s` , the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, actual_rank, str(tensor.shape), str(expected_rank)))


if __name__ == "__main__":
    features = tf.initializers.TruncatedNormal()(shape=[8, 784])
    a = tf.keras.layers.LayerNormalization(epsilon=0.00001)(features)
    print(a)
