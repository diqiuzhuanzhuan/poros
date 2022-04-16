# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

import pytorch_lightning as pl
import os
import time
import logging
import gc


def wait_gc(wait=5):
    count = 0
    while count <= wait:
        time.sleep(1)
        gc.collect()
        count += 1


def save_model(trainer: pl.Trainer, out_dir, model_name='', timestamp=None):
    out_dir = out_dir + '/lightning_logs/version_' + str(trainer.logger.version) + '/checkpoints/'
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)

    outfile = out_dir + '/' + model_name + '_timestamp_' + str(timestamp) + '_final.ckpt'
    trainer.save_checkpoint(outfile, weights_only=True)

    logging.info('Stored model {}.'.format(outfile))
    best_checkpoint = None
    for file in os.listdir(out_dir):
        if file.startswith("epoch"):
            best_checkpoint = os.path.join(out_dir, file)
            break
    return outfile, best_checkpoint