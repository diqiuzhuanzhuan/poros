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


def load_model(model_class: pl.LightningModule, model_file, stage='test', **kwargs):
    hparams_file = model_file[:model_file.rindex('checkpoints/')] + '/hparams.yaml'
    model = model_class.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, **kwargs)
    model.stage = stage
    return model


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

    
def write_eval_performance(eval_performance, out_file):
    outstr = ''
    added_keys = set()
    for out_ in eval_performance:
        for k in out_:
            if k in added_keys or k in ['results', 'predictions']:
                continue
            outstr = outstr + '{}\t{}\n'.format(k, out_[k])
            added_keys.add(k)

    open(out_file, 'wt').write(outstr)
    logging.info('Finished writing evaluation performance for {}'.format(out_file))