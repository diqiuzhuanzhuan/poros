# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import torch


class BiaffineLayer(torch.nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.zeros(
            in_size + int(bias_x), out_size, in_size + int(bias_y)))
        torch.nn.init.normal_(self.U, mean=0, std=0.1)

    def forward(self, x, y):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: a tensor, shape is [batch, x, y, o]
        """
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping
