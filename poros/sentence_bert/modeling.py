# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


import transformers
from transformers import BertTokenizer
import tensorflow as tf


class ClsLayer(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
       super(ClsLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        return inputs[:, 0:1, :]


class SiameseLayer(tf.keras.layers.Layer):

    def __init__(self, pretrain_name="bert-base-uncased", pool_method="avg", **kwargs):
        super(SiameseLayer, self).__init__(**kwargs)
        self.pretrain_name = pretrain_name
        self.pool_method = pool_method
        self.bert_model = transformers.TFBertModel.from_pretrained(pretrained_model_name_or_path=pretrain_name)
        self.bert_config = self.bert_model.config.from_pretrained(self.pretrain_name).to_dict()
        if self.pool_method == 'avg':
            self.pool_layer = tf.keras.layers.AvgPool2D([self.bert_config["max_position_embeddings"], 1])
        elif self.pool_method == 'max':
            self.pool_layer = tf.keras.layers.MaxPool2D([self.bert_config["max_position_embeddings"], 1])
        elif self.pool_method == 'cls':
            self.pool_layer = ClsLayer()
        else:
            raise ValueError("don't support {}".format(self.pool_method))

    def get_config(self):
        return self.bert_config

    def call(self, inputs_a, inputs_b=None):
        """

        :param inputs_a:
        :param inputs_b:
        :return:
        """
        outputs_a = self.bert_model(**inputs_a)
        # pool_inputs_a: [batch, seq_length, hidden]
        pool_inputs_a = outputs_a["last_hidden_state"]
        pool_inputs_a = tf.expand_dims(pool_inputs_a, 3)
        pool_outputs_a = self.pool_layer(pool_inputs_a)
        pool_outputs_a = tf.reshape(pool_outputs_a, [-1, self.bert_config["hidden_size"]])
        if inputs_b:
            outputs_b = self.bert_model(**inputs_b)
            pool_inputs_b = outputs_b["last_hidden_state"]
            pool_inputs_b = tf.expand_dims(pool_inputs_b, 3)
            pool_outputs_b = self.pool_layer(pool_inputs_b)
            pool_outputs_b = tf.reshape(pool_outputs_b, [-1, self.bert_config["hidden_size"]])
            return pool_outputs_a, pool_outputs_b
        return pool_outputs_a


if __name__ == "__main__":
    sbl = SiameseLayer(pool_method='avg')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs_a = tokenizer.tokenize("Hello, my dog is gone, can you help me to find it?", return_tensors='tf')
    inputs_b = tokenizer("Hello, my cat is gone, can you help me to find it?", return_tensors='tf')
    print(inputs_a)
    outputs = sbl(inputs_a, inputs_b)
    print(outputs*3)
    outputs = sbl(inputs_a)
