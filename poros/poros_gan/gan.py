# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os


class GanModel(object):

    def __init__(self):
        self.noise_dimension = 100
        self.batch_size = 512
        self.generator_layer_units = [256, 256, 256, 784]
        self.discriminator_layer_units = [256, 128]
        (self.train_x, self.train_y), (_, _) = keras.datasets.mnist.load_data()
        self.train_x = self.train_x.reshape([self.train_x.shape[0], 784]).astype("float32")
        self.train_x = (self.train_x - 255/2)/(255/2)

        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_x).shuffle(buffer_size=self.batch_size * 50).batch(self.batch_size)

    def make_generator_model(self):
        # 定义生成器
        self.generator_model = keras.Sequential()
        self.generator_model.add(keras.layers.Dense(units=self.generator_layer_units[0], use_bias=True, input_shape=(100, )))
        self.generator_model.add(keras.layers.BatchNormalization())
        self.generator_model.add(keras.layers.LeakyReLU())

        assert self.generator_model.output_shape == (None, 256)

        self.generator_model.add(keras.layers.Dense(units=self.generator_layer_units[1], use_bias=True))
        self.generator_model.add(keras.layers.BatchNormalization())
        self.generator_model.add(keras.layers.LeakyReLU())

        assert self.generator_model.output_shape == (None, 256)

        self.generator_model.add(keras.layers.Dense(units=self.generator_layer_units[2], use_bias=True))
        self.generator_model.add(keras.layers.BatchNormalization())
        self.generator_model.add(keras.layers.LeakyReLU())

        assert self.generator_model.output_shape == (None, 256)

        self.generator_model.add(keras.layers.Dense(units=self.generator_layer_units[3], use_bias=True, activation='tanh'))
        assert self.generator_model.output_shape == (None, 784)

        self.generator_model.add(keras.layers.Reshape((28, 28)))
        assert self.generator_model.output_shape == (None, 28, 28)

        return self.generator_model

    def make_discriminator_model(self):
        # 定义判别器
        self.discriminator_model = keras.Sequential()
        self.discriminator_model.add(keras.layers.Reshape((784,), input_shape=(28, 28)))
        self.discriminator_model.add(keras.layers.Dense(units=self.discriminator_layer_units[0], use_bias=True))
        self.discriminator_model.add(keras.layers.BatchNormalization())
        self.discriminator_model.add(keras.layers.LeakyReLU())
        self.discriminator_model.add(keras.layers.Dropout(0.4))
        assert self.discriminator_model.output_shape == (None, 256)

        self.discriminator_model.add(keras.layers.Dense(units=self.discriminator_layer_units[1], use_bias=True))
        self.discriminator_model.add(keras.layers.BatchNormalization())
        self.discriminator_model.add(keras.layers.LeakyReLU())
        self.discriminator_model.add(keras.layers.Dropout(0.3))
        assert self.discriminator_model.output_shape == (None, 128)

        self.discriminator_model.add(keras.layers.Dense(units=1, activation="sigmoid"))
        assert self.discriminator_model.output_shape == (None, 1)

        return self.discriminator_model

    def make_generator_loss(self, input):
        # 定义生成器的损失函数
        binary_cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        return binary_cross_entropy(y_true=tf.ones_like(input), y_pred=input)

    def make_generator_optimizer(self):
        # 定义生成器的优化器
        self.generator_optimizer = keras.optimizers.Adam(1e-4)
        return self.generator_optimizer

    def make_discriminator_optimizer(self):
        # 定义判别器的优化器
        self.discriminator_optimizer = keras.optimizers.Adam(1e-4)
        return self.discriminator_optimizer

    def make_discriminator_loss(self, real, fake):
        # 定义判别器的损失函数
        binary_cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = binary_cross_entropy(tf.ones_like(real), real)
        fake_loss = binary_cross_entropy(tf.zeros_like(fake), fake)
        return real_loss + fake_loss

    def make_checkpoint_saver(self):
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator_model,
                                         discriminator=self.discriminator_model,
                                        )

    def build(self):
        self.make_generator_model()
        self.make_discriminator_model()
        self.make_generator_optimizer()
        self.make_discriminator_optimizer()
        self.make_checkpoint_saver()

    def eval_generator(self):
        noise = tf.random.normal([9, self.noise_dimension])
        image = self.generator_model(noise, training=False)
        precision = self.discriminator_model(image, training=False)
        tf.get_logger().info("presion is\n {}".format(precision))
        image = image * 255/2 + 255/2
        fig = plt.figure(figsize=(3, 3))
        for i in range(image.shape[0]):
            plt.subplot(3, 3, i+1)
            plt.imshow(image[i, :, :], cmap='gray')
            plt.axis("off")
        plt.show()

    @tf.function
    def train_step(self, images):
        # 使用tf.function装饰器来实现autograph
        noise = tf.random.normal([self.batch_size, self.noise_dimension])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generative_image = self.generator_model(noise, training=True)
            real_output = self.discriminator_model(images, training=True)
            fake_output = self.discriminator_model(generative_image, training=True)

            self.gen_loss = self.make_generator_loss(fake_output)

            self.disc_loss = self.make_discriminator_loss(real_output, fake_output)

        gen_gradient = gen_tape.gradient(self.gen_loss, self.generator_model.trainable_variables)
        disc_gradient = disc_tape.gradient(self.disc_loss, self.discriminator_model.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_gradient, self.generator_model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradient, self.discriminator_model.trainable_variables))
        return self.gen_loss, self.disc_loss

    def train(self, epochs):
        train_steps = 0
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        for epoch in range(epochs):
            self.train_dataset.repeat(1)
            for batch_data in self.train_dataset:
                gen_loss, disc_loss = self.train_step(batch_data)
                tf.get_logger().info("gen_loss is {}, disc_loss is {}".format(gen_loss, disc_loss))
                train_steps += 1
                if train_steps % 10000 == 0:
                    self.eval_generator()
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)


if __name__ == "__main__":
    import logging
    tf.get_logger().setLevel(logging.INFO)
    model = GanModel()
    model.build()
    model.train(epochs=500)
