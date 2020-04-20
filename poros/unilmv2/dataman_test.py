# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import unittest
from .dataman import Sample


class SampleTest(unittest.TestCase):

    def setUp(self) -> None:
        self.sample = Sample()

    def test_block_wise_masking(self):
        x = ["SOS", "你", "好", "我", "是", "20", "##19", "一", "个", "好",
             "的", "人", "EOS", "你", "呢", "你", "在", "干", "嘛", "呢", "EOS"]
        M = self.sample.block_wise_masking(x, max_predictions_per_seq=0, mask_ratio=0.15)
        self.assertFalse("[MASK]" in M[0])
        M = self.sample.block_wise_masking(x, max_predictions_per_seq=1, mask_ratio=0.15)
        self.assertTrue("[MASK]" in M[0])
        print(M)




