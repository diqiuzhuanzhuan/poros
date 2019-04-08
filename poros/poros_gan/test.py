# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
from poros.poros_gan import gan
import tensorflow as tf
import matplotlib.pyplot as plt


def test_make_generator():
    model = gan.GanModel()
    generator = model.make_generator_model()
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :])
    plt.show()


def test_make_discriminator():
    model = gan.GanModel()
    generator = model.make_generator_model()
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    discriminator = model.make_discriminator_model()
    discriminator_score = discriminator(generated_image, training=False)
    print(discriminator_score)


if __name__ == "__main__":
    test_make_generator()
    test_make_discriminator()