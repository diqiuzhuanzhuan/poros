# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import pytorch_lightning as pl
from typing import Optional
import torch.utils.data


class CFDataModule(pl.LightningDataModule):

    def __init__(self, train_data, validation_data=None, test_data=None):
        super(CFDataModule, self).__init__()
        self.train_data = train_data
        self.validation_data = validation_data or train_data
        self.test_data = test_data or train_data

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_data, batch_size=1)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=1)


if __name__ == "__main__":
    import numpy as np
    train_data = np.random.random_integers(0, 1, size=[10, 4])
    cfdm = CFDataModule([train_data])
    for i in cfdm.train_dataloader():
        print(i)

