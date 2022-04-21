# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from typing import List, Any
import os
import pytorch_lightning.core.lightning as pl
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModelForTokenClassification
import logging
from poros_metrics.span_metric import SpanF1
from reader_utils import extract_spans, get_tags


class NERBaseAnnotator(pl.LightningModule):
    def __init__(self,
                 train_data=None,
                 dev_data=None,
                 test_data=None,
                 lr=1e-5,
                 dropout_rate=0.1,
                 batch_size=16,
                 tag_to_id=None,
                 stage='fit',
                 pad_token_id=1,
                 encoder_model='xlm-roberta-large',
                 num_gpus=1,
                 use_crf=False,
                 kl_loss_config=[('O', 'B-PROD', 1.5),
                                 ('O', 'I-PROD', 1.5),
                                 ('O', 'B-CW', 1),
                                 ('O', 'I-CW', 1),
                                 ],
                 l2_loss_config=[(1, 'B-PROD'), 
                                 (1, 'I-PROD'), 
                                 (0.8, 'B-CW'), 
                                 (0.8, 'I-CW'), 
                                 (0.3, 'B-PER'), 
                                 (0.3, 'I-PER')
                                 ],
                 alpha=0.3
                 ):
        super(NERBaseAnnotator, self).__init__()

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.batch_size = batch_size

        self.stage = stage
        self.num_gpus = num_gpus
        self.target_size = len(self.id_to_tag)
        self.use_crf = use_crf
        self.kl_loss_config = kl_loss_config
        self.l2_loss_config = l2_loss_config

        self.alpha = alpha

        # set the default baseline model here
        self.pad_token_id = pad_token_id

        self.encoder_model = encoder_model
        try:
            self.encoder = AutoModelForTokenClassification.from_pretrained(encoder_model, num_labels=self.target_size, classifier_dropout=dropout_rate)
        except TypeError as e:
            self.encoder = AutoModelForTokenClassification.from_pretrained(encoder_model, num_labels=self.target_size)
            logging.warning("{} has no classifier_dropout parameter".format(encoder_model))
        if self.use_crf:
            self.crf_layer = ConditionalRandomField(num_tags=self.target_size, constraints=allowed_transitions(constraint_type="BIO", labels=self.id_to_tag))
        
        self.auxiliary_classifier = nn.Linear(self.encoder.config.hidden_size, 2)

        # adversial
        self.emb_backup = {}
        self.grad_backup = {}

        self.lr = lr
        self.span_f1 = SpanF1()
        self.val_span_f1 = SpanF1()
        self.setup_model(self.stage)
        self.save_hyperparameters('pad_token_id', 'encoder_model', 'use_crf')

        self.test_result = []
        self.val_result = []

    def setup_model(self, stage_name):
        if stage_name == 'fit' and self.train_data is not None:
            # Calculate total steps
            train_batches = len(self.train_data) // (self.batch_size * self.num_gpus)
            self.total_steps = 50 * train_batches

            self.warmup_steps = int(self.total_steps * 0.01)

    
    def collate_batch(self, mode='val'):
        if mode == 'fit':
            return self.train_collate_batch
        else:
            return self.train_collate_batch

    def train_collate_batch(self, batch):
        batch_ = list(zip(*batch))
        tokens, masks, gold_spans, tags, subtoken_pos_to_raw_pos, token_type_ids, position_ids = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4], batch_[5], batch_[6]

        max_len = max([len(token) for token in tokens])
        token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(self.pad_token_id)
        # -100 is the default ignore index
        if self.use_crf:
            ignore_index = 0
        else:
            ignore_index = -100
        tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(ignore_index)
        mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
        mask_tensor = torch.zeros(size=(len(tokens), max_len, max_len), dtype=torch.bool)
        tag_len_tensor = torch.zeros(size=(len(tokens),), dtype=torch.long)
        token_type_ids_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(0)
        auxiliary_tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(-100) # don't modify this -100, as it is not belong to ner tag

        for i in range(len(tokens)):
            tokens_ = tokens[i]
            seq_len = len(tokens_)
            tag_len = len(tags[i])
            token_tensor[i, :seq_len] = tokens_
            tag_len_tensor[i] = tag_len
            
            tag_tensor[i, :tag_len] = tags[i]
            auxiliary_tag_tensor[i, :tag_len] = torch.tensor([0 if self.id_to_tag[j.item()] == 'O' else 1 for j in tags[i]])
            mask_tensor[i, :seq_len, :seq_len] = masks[i]
            token_type_ids_tensor[i, :seq_len] = token_type_ids[i]

        return token_tensor, tag_tensor, mask_tensor, token_type_ids_tensor, gold_spans, subtoken_pos_to_raw_pos, tag_len_tensor, auxiliary_tag_tensor

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        if self.stage == 'fit':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps)
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        return [optimizer]

    def train_dataloader(self):
        loader = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_batch(mode='train'), num_workers=os.cpu_count(), shuffle=True)
        return loader

    def test_dataloader(self):
        if self.test_data is None:
            return None
        loader = DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.collate_batch(mode='val'), num_workers=os.cpu_count(), shuffle=False)
        return loader

    def val_dataloader(self):
        if self.dev_data is None:
            return None
        loader = DataLoader(self.dev_data, batch_size=self.batch_size, collate_fn=self.collate_batch(mode='val'), num_workers=os.cpu_count())
        return loader

    def test_epoch_end(self, outputs):
        pass

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, suffix='', on_step=False, on_epoch=True)
        self.span_f1.reset()

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pred_results = self.val_span_f1.get_metric(True)
        avg_loss = np.mean([preds['loss'].item() for preds in outputs])
        self.log_metrics(pred_results, loss=avg_loss, suffix='val_', on_step=False, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        self.val_result = []
        self.val_span_f1.reset()
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch, mode='val')
        self.log_metrics(output['results'], loss=output['loss'], suffix='val_', on_step=True, on_epoch=False)
        [self.val_result.append(res) for res in output['raw_token_results']]
        return output

    def training_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch, mode='fit')
        self.log_metrics(output['results'], loss=output['loss'], suffix='', on_step=True, on_epoch=False)
        return output

    def training_step_end(self, *args, **kwargs):

        return super().training_step_end(*args, **kwargs)
    
    def on_test_epoch_start(self) -> None:
        self.test_result = []
        return super().on_test_epoch_start()
    

    def test_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch, mode=self.stage)
        [self.test_result.append(res) for res in output['raw_token_results']]
        return output

    def log_metrics(self, pred_results, loss=0.0, suffix='', on_step=False, on_epoch=True):
        for key in pred_results:
            self.log(suffix + key, pred_results[key], on_step=on_step, on_epoch=on_epoch, prog_bar=True, logging=True)

        self.log(suffix + 'loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logging=True)
    
    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        for name, param in self.encoder.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)
    
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.encoder.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

    def perform_forward_step(self, batch, mode=''):
        tokens, tags, token_mask, token_type_ids, metadata, subtoken_pos_to_raw_pos, tag_len, auxiliary_tag = batch
        batch_size = tokens.size(0)

        outputs = self.encoder(input_ids=tokens, attention_mask=token_mask, labels=tags, output_hidden_states=True)
        hidden_states = outputs.hidden_states[0]
        auxiliary_logits = self.auxiliary_classifier(hidden_states)

        loss_fct = CrossEntropyLoss()
        auxiliary_loss = loss_fct(auxiliary_logits.view(-1, 2), auxiliary_tag.view(-1))
        # compute the log-likelihood loss and compute the best NER annotation sequence
        token_scores = outputs.logits
        kl_loss = self._add_kl_loss()
        l2_loss = self._add_l2_regulization(l2_config=self.l2_loss_config)
        alpha = np.exp(-self.global_step/10000)
        loss = (1-self.alpha) *  outputs.loss + self.alpha * auxiliary_loss - kl_loss + l2_loss

        output = self._compute_token_tags(token_scores=token_scores, tags=tags, token_mask=token_mask, 
                                          metadata=metadata, subtoken_pos_to_raw_pos=subtoken_pos_to_raw_pos, batch_size=batch_size, mode=mode, tag_lens=tag_len)
        if not output['loss']:
            output['loss'] = loss
        return output

    def _add_kl_loss(self):
        loss = 0.0
        for tag1, tag2, threshold in self.kl_loss_config:
            loss_fct = torch.nn.KLDivLoss(log_target=True)
            tag1_idx, tag2_idx = self.tag_to_id[tag1], self.tag_to_id[tag2]
            _loss = loss_fct(self.encoder.classifier.weight[tag1_idx], self.encoder.classifier.weight[tag2_idx])
            loss += min(_loss - threshold, 0)
        return loss
    
    def _add_l2_regulization(self, l2_config):
        loss = 0.0
        for weight, tag in l2_config:
            loss_fct = torch.nn.MSELoss()
            target = self.encoder.classifier.weight[self.tag_to_id[tag]].clone().detach()
            loss += weight * loss_fct(self.encoder.classifier.weight[self.tag_to_id[tag]], target)
        return loss

    def _compute_token_tags(self, token_scores, tags, token_mask, metadata, subtoken_pos_to_raw_pos, batch_size, tag_lens, mode=''):
        if self.use_crf:
        # compute the log-likelihood loss and compute the best NER annotation sequence
            # we need to modify -100 to 0, for the sake of running normaly in crf function
            """
            if token_mask.dim() == 3:
                crf_token_mask = (tags != -100) & token_mask[:, 0]
            else:
                crf_token_mask = (tags != -100) & token_mask
            """
            crf_token_mask = torch.empty(size=tags.size(), dtype=torch.bool).fill_(False).to(self.device)
            for i, tag_len in enumerate(tag_lens):
                crf_token_mask[i, :tag_len] = True
            loss = -self.crf_layer(token_scores, tags, crf_token_mask) / float(batch_size)
            best_path = self.crf_layer.viterbi_tags(token_scores, crf_token_mask)
        else:
            loss = None
            best_path = torch.argmax(token_scores, -1)

        pred_results = []
        raw_pred_results = []
        raw_token_results = []
        for i in range(batch_size):
            tag_len = tag_lens[i].item()
            if self.use_crf:
                tag_seq, _ = best_path[i]
                tag_seq = tag_seq[:tag_len]
            else:
                tag_seq = best_path[i].cpu().numpy()[0:tag_len]
            span_res, raw_token_res = extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag], subtoken_pos_to_raw_pos[i])
            pred_results.append(span_res)
            raw_token_results.append(raw_token_res)
            raw_pred_results.append([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag])
        output = {"loss": loss, "pred_results": pred_results, "raw_pred_results": raw_pred_results, 'raw_token_results': raw_token_results}
        if mode == 'val':
            self.val_span_f1(pred_results, metadata)
            output["results"] = self.val_span_f1.get_metric()
        elif mode == 'test':
            pass
        else:
            self.span_f1(pred_results, metadata)
            output["results"] = self.span_f1.get_metric()

        return output

    def predict_tags(self, batch, tokenizer=None):
        tokens, tags, token_mask, metadata = batch
        pred_tags = self.perform_forward_step(batch, mode='predict')['token_tags']
        token_results, tag_results = [], []
        for i in range(tokens.size(0)):
            instance_token_results, instance_tag_results = get_tags(tokens[i], pred_tags[i], tokenizer=tokenizer)
            token_results.append(instance_token_results)
            tag_results.append(instance_tag_results)
        return token_results, tag_results
