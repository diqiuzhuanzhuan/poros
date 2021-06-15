# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import tensorflow as tf
from poros.sentence_bert import SiameseLayer
import tensorflow_addons as tfa
from transformers import BertTokenizer


class TransparentLayer(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super(TransparentLayer, self).__init__(*args, **kwargs)

    def call(self, input_a, input_b, **kwargs):
        input_a_l2 = tf.nn.l2_normalize(input_a, axis=-1)
        input_b_l2 = tf.nn.l2_normalize(input_b, axis=-1)
        return sum(input_a_l2 * input_b_l2)


class DistanceLayer(tf.keras.layers.Layer):
    
    def __init__(self, units, *args, **kwargs):
        super(DistanceLayer, self).__init__(*args, **kwargs)
        self.units = units
        self.weight = None

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("input shape must be a tuple of TensorShape")
        self.weight = self.add_weight(shape=(input_shape[-1]*3, self.units),
                                      initializer='random_normal',
                                      trainable=True)

    def call(self, input_a, input_b, **kwargs):
        concatenate_input = tf.keras.layers.concatenate([input_a, input_b, tf.abs(input_a-input_b)])
        return tf.matmul(concatenate_input, self.weight)


class SentenceBert(tf.keras.Model):

    def __init__(self,
                 pretrain_name="bert-base-uncased",
                 pool_method="avg",
                 loss_fn="softmax",
                 *args,
                 **kwargs):
        """

        :param pretrain_name:
        :param loss_fn: softmax if classifier or mse if regression
        :param args:
        :param kwargs:
        """

        super(SentenceBert, self).__init__(*args, **kwargs)
        self.siamese_layer = SiameseLayer(pretrain_name=pretrain_name, pool_method=pool_method)
        self.bert_config = self.siamese_layer.get_config()
        self.loss_name = loss_fn
        if loss_fn == 'softmax':
            self.output_layer = DistanceLayer(units=self.bert_config["hidden_size"])
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        elif loss_fn == 'regression':
            self.output_layer = TransparentLayer()
            self.loss_fn = tf.losses.MSE()
        elif loss_fn == 'triple_loss':
            self.output_layer = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
            self.loss_fn = tfa.losses.TripletSemiHardLoss()
        else:
            raise ValueError("don't support {}".format(loss_fn))
        self.loss_metric = tf.keras.metrics.Mean(name=loss_fn)

    def call(self, inputs, training):
        # 如果是输入两个句子
        if inputs.get('input_a', None):
            input_a = inputs['input_a']
            input_b = inputs['input_b']
            output_a = self.siamese_layer(input_a)
            output_b = self.siamese_layer(input_b)
            outputs = self.output_layer(output_a, output_b)
        else:
            siamese_output = self.siamese_layer(inputs)
            outputs = self.output_layer(siamese_output)

        if training:
            label_ids = inputs["label_ids"]
            loss = self.loss_fn(y_true=label_ids, y_pred=outputs)
            self.add_loss(loss)
            loss_metric = self.loss_metric(loss)
            self.add_metric(loss_metric)
            return outputs, loss
        return outputs


if __name__ == "__main__":
    from poros.sentence_bert.dataman import SnliDataMan
    snli_dataman = SnliDataMan()
    data = snli_dataman.batch(data_type='train', batch_size=32)
    sbm = SentenceBert(loss_fn='triple_loss')
    optimazer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    sbm.compile(optimizer=optimazer)
    sbm.fit(data, epochs=1, steps_per_epoch=20)