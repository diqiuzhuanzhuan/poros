# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

import tensorflow as tf
from poros.poros_train import some_layer
from poros.poros_train import optimization
from poros.poros_dataset import about_tensor


class TextCNN(tf.keras.Model):

    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 filter_size,
                 num_filters,
                 poll_out_drop=0.1,
                 l2_reg_lambda=0.0):
        super(TextCNN, self).__init__()
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_size
        self.num_filters = num_filters
        self.poll_out_drop = poll_out_drop
        self.l2_reg_lambda = l2_reg_lambda
        self.num_filters_total = self.num_filters * len(self.filter_sizes)

        self.embedding_layer = some_layer.EmbeddingLookupLayer(vocab_size=self.vocab_size, embedding_size=self.embedding_size)
        self.filter_W = []
        self.filter_b = []
        self.l2_loss = 0

        self.pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(
                    initial_value=tf.initializers.TruncatedNormal(stddev=0.1)(shape=filter_shape),
                    name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.filter_W.append(W)
                self.filter_b.append(b)

        with tf.name_scope("output"):
            self.output_W = tf.Variable(
                initial_value=tf.initializers.TruncatedNormal(stddev=0.02)(shape=[self.num_filters_total, self.num_classes]),
                name="w"
            )
            self.output_b = tf.Variable(
                initial_value=tf.initializers.Constant(0.1)(shape=[self.num_classes]),
                name="b"
            )
        with tf.name_scope("dropout"):
            self.dropout_layer = tf.keras.layers.Dropout(rate=self.poll_out_drop)

        with tf.name_scope("metrics"):
            self.accuracy_metric = tf.keras.metrics.Accuracy()

    def call(self, inputs, training=False):

        input_ids = inputs["input_ids"]
        input_y = inputs["input_y"]
        print(input_y)
        input_y = tf.one_hot(input_y, depth=self.num_classes)
        print(input_y)
        embedding_output, _ = self.embedding_layer(input_ids)
        # now the dimension is [B, H, W], so we expand it to [B, H, W, C]
        expanded_embedding_output = tf.expand_dims(embedding_output, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                expanded_embedding_output,
                self.filter_W[i],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool'
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.filter_b[i]), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="pool"
            )
            pooled_outputs.append(pooled)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
        h_drop = self.dropout_layer(self.h_pool_flat)

        self.l2_loss = tf.nn.l2_loss(self.output_W)
        self.l2_loss += tf.nn.l2_loss(self.output_b)
        scores = tf.compat.v1.nn.xw_plus_b(h_drop, self.output_W, self.output_b, name="scores")
        predictions = tf.argmax(scores, -1, name="predictions")

        # Calculate mean cross-entropy loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
        total_loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
        self.add_loss(total_loss)

        accuracy = self.accuracy_metric(predictions, tf.argmax(input_y, -1))
        if not self.build:
            self.add_metric(accuracy)
        self.summary()

        return predictions


if __name__ == "__main__":
    import numpy as np
    model = TextCNN(sequence_length=6, num_classes=2, vocab_size=100, embedding_size=128, filter_size=[2, 3, 4],
                    num_filters=3, l2_reg_lambda=0.1)
    inputs = {
        "input_ids": np.array([
            [1, 2, 3, 4, 5, 6],
            [1, 3, 3, 7, 5, 6],
            [1, 6, 3, 7, 4, 6],
            [1, 3, 5, 7, 5, 6],
            [1, 3, 1, 7, 5, 6]
        ]),
        "input_y": np.array([0, 1, 0, 1, 0])
    }

    print(model(inputs))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer)
    data = tf.data.Dataset.from_tensor_slices(inputs)
    data = data.repeat().batch(5, drop_remainder=True)
    model.fit(data, epochs=1000, steps_per_epoch=1)
