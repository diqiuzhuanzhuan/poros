# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import torch

class GravityLoss(torch.nn.modules.loss._Loss):

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(GravityLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input_u: torch.Tensor, input_v: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param input_v:
        :param input_u:
        :param target:
        :return:
        """
        if torch.Size([input_u.shape[0], input_v.shape[0]]) != target.size():
            raise ValueError(
                "Using a target size ({}) that is different to the input size ({}) is deprecated. "
                "Please ensure they have the same size.".format(target.size(), [input_u.shape[0] * input_v.shape[0]])
            )

        if self.reduction == 'mean':
            m = input_u.shape[0]
            n = input_v.shape[0]
        else:
            m = n = 1
        loss = 1.0 / (m*n) * torch.matrix_power(torch.matmul(input_u, input_v.t()), 2)
        return loss


class DiceLoss(torch.nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(DiceLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, target: torch.Tensor, prediction: torch.Tensor, ep=1e-8) -> torch.Tensor:
        intersection = 2 * torch.sum(target * prediction) + ep
        union = torch.sum(prediction) + torch.sum(target) + ep
        loss = 1 - intersection/union
        return loss
