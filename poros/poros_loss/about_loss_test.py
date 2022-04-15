# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

import unittest
from poros.poros_loss.about_loss import GravityLoss
import torch


class ModelingTest(unittest.TestCase):

    def test_gravity_loss(self):
        gl = GravityLoss()
        # [1, 2]
        input_a = torch.tensor([[1.0, 1]], requires_grad=True)
        input_b = torch.tensor([[1.0, 1]], requires_grad=True)
        target = torch.tensor([[4.0]])
        output = gl(input_a, input_b, target)
        torch.testing.assert_close(output, target)


if __name__ == "__main__":
    unittest.main()