# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import torch
import pytorch_lightning as pl
from poros.poros_loss import about_loss


class RegularLoss(torch.nn.modules.loss._Loss):

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(RegularLoss, self).__init__(size_average, reduce, reduction)
        self.user_regular_loss = torch.nn.MSELoss()
        self.item_regular_loss = torch.nn.MSELoss()

    def forward(self, input_u: torch.Tensor, input_v: torch.Tensor) -> torch.Tensor:
        """
        :param input_v:
        :param input_u:
        :return:
        """
        user_regular_loss = self.user_regular_loss(input_u, torch.zeros(size=input_u.shape))
        item_regular_loss = self.item_regular_loss(input_v, torch.zeros(size=input_v.shape))
        return user_regular_loss + item_regular_loss


class CFModel(pl.LightningModule):

    def __init__(self, user_num, item_num, embedding_dim, learning_rate=10, lambda_gravity=0.3, lambda_regular=0.4):
        super(CFModel, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=embedding_dim)
        self.mse_loss = torch.nn.MSELoss()
        self.gravity_loss = about_loss.GravityLoss()
        self.regular_loss = RegularLoss()
        self.learning_rate = learning_rate
        self.lambda_gravity = lambda_gravity
        self.lambda_regular = lambda_regular

    def forward(self, user_id, item_id):
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        return user_embedding, item_embedding

    def training_step(self, target_matrix: torch.Tensor, idx):
        y = torch.matmul(self.user_embedding.weight, self.item_embedding.weight.t())
        mse_loss = self.mse_loss(y, target_matrix)
        gravity_loss = self.gravity_loss(self.user_embedding.weight, self.item_embedding.weight)
        regular_loss = self.regular_loss(self.user_embedding.weight, self.item_embedding.weight)
        loss = mse_loss + self.lambda_gravity * gravity_loss + self.lambda_regular * regular_loss
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, target_matrix, idx):
        y = torch.matmul(self.user_embedding.weight, self.item_embedding.weight.t())
        error = torch.nn.MSELoss()(y, target_matrix)
        self.log("validation error", error, prog_bar=True)

    def test_step(self, target_matrix, idx):
        y = torch.matmul(self.user_embedding.weight, self.item_embedding.weight.t())
        error = torch.nn.MSELoss()(y, target_matrix)
        self.log("test error", error, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    pass