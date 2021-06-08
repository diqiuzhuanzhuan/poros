# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import unittest
import tensorflow as tf
from sentence_bert.dataman import SnliDataMan


class SnliDataManTest(unittest.TestCase):

    def setUpClass(cls) -> None:
        cls.snli_dataman = SnliDataMan()

    def test_gen(self):
        data = tf.data.Dataset.from_generator(self.snli_dataman.gen(data_type='train'), output_types=())



