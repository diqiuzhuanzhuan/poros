# coding=utf-8
# Copyright 2020 diqiuzhuanzhuan Authors.
# email: diqiuzhuanzhuan@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

import tensorflow as tf
import matplotlib as plt


class GradientPrint(tf.keras.callbacks.Callback):

    def __init__(self, loss, variables):
        super(GradientPrint, self).__init__()
        self.loss = loss
        self.variables = variables

    def on_batch_end(self, batch, logs=None):
        gradient = tf.keras.backend.gradients(self.loss, self.variables)
        tf.print(gradient)


def imshow_zero_center(image, **kwargs):
    lim = tf.reduce_max(abs(image))
    plt.imshow(image, vmin=-lim, vmax=lim, cmap='seismic', **kwargs)
    plt.colorbar()
