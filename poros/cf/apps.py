# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import pytorch_lightning as pl
from poros.cf import modeling
from poros.cf import dataman
import numpy as np
import torch

if __name__ == "__main__":
    AVAIL_GPUS = torch.cuda.device_count() or None
    train_data = np.random.random_integers(0, 1, size=[10, 4])
    train_data = np.float32(train_data)
    cfdm = dataman.CFDataModule([train_data])
    model = modeling.CFModel(10, 4, 10, learning_rate=0.1)
    trainer = pl.Trainer(
        max_epochs=1000,
        progress_bar_refresh_rate=5,
        gpus=AVAIL_GPUS,
        log_every_n_steps=1,
        val_check_interval=1.0
    )
    trainer.fit(model=model, datamodule=cfdm)
    print(torch.matmul(model.user_embedding.weight, model.item_embedding.weight.t()))
    print(train_data)