# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import tensorflow as tf
from poros.unilmv2.config import Unilmv2Config
from poros.unilmv2 import Unilmv2Layer


class Unilmv2Model(tf.keras.Model):

    def __init__(self, config: Unilmv2Config, is_training=False, **kwargs):
        if not isinstance(config, Unilmv2Config):
            raise TypeError("config type muse be Unilmv2Config")
        super(Unilmv2Model, self).__init__(**kwargs)
        self.config = config
        self.unilmv2_layer = Unilmv2Layer(config=self.config, is_training=is_training)

    def call(self, inputs, training=None, mask=None):
        self.unilmv2_layer(inputs)


if __name__ == "__main__":
    pass



