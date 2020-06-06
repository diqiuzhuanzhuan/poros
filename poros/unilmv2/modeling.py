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
)
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


class AutoEncodingModel(tf.keras.Model):

    def __init__(self, config: Unilmv2Config):
        super(AutoEncodingModel, self).__init__()
        if isinstance(config, Unilmv2Config):
            raise TypeError("config type muse be Unilmv2Config")
        self.transformer_layer = TransformerLayer(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob)


