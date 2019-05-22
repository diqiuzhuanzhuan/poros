# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
from poros.poros_train import some_layer


class SomeLayerTest(tf.test.TestCase):

    def test_dot_product_attention(self):

        with self.test_session() as sess:
            query = tf.get_variable("w", shape=[3, 4], initializer=tf.constant_initializer(
                [
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                ]
            ))

            out = some_layer.dot_product_attention(q=query, k=query, v=query, scale=True, bias=None)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            out = sess.run(out)
            out_ = tf.constant([
                [2.85093709, 2.85093709, 2.85093709, 2.85093709],
                [2.98136107, 2.98136107, 2.98136107, 2.98136107],
                [2.99751513, 2.99751513, 2.99751513, 2.99751513]
            ]
            )
            self.assertAllClose(out, out_)


if __name__ == "__main__":
    tf.test.main()
