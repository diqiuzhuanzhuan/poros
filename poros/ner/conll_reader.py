# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from collections import defaultdict
import torch
from torch.utils.data import Dataset
import random
from transformers import AutoTokenizer, LukeTokenizer
import copy
import ahocorasick
from intervaltree import IntervalTree, Interval
from poros.ner.reader_utils import get_ner_reader, extract_spans, _assign_ner_tags
import os
import itertools
import numpy as np
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CoNLLReader(Dataset):
    def __init__(self, min_instances=0, max_instances=-1, max_length=50, target_vocab=None, pretrained_dir='', encoder_model='xlm-roberta-large', entity_vocab: dict = None):
        self._max_instances = max_instances
        self._min_instances = min_instances
        self._max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model)
        self.entity_vocab = entity_vocab
        if self.entity_vocab:
            self._setup_entity_vocab()

        self.cls_token = self.tokenizer.special_tokens_map['cls_token']
        self.cls_token_id = self.tokenizer.get_vocab()[self.cls_token]
        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']
        self.sep_token_id = self.tokenizer.get_vocab()[self.sep_token]

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []
        # split into many pieces, ease --> e ase
        self.word_piece_ids = []
        self.pos_to_single_word_maps = []
        self.ner_tags = []
        self.type_count = defaultdict(int)
        self.type_to_entityset = defaultdict(set)
        self.label_entities = defaultdict(int)
    
    def _setup_entity_vocab(self):
        self.entity_automation = ahocorasick.Automaton()
        tmp = dict()
        for k in self.entity_vocab:
            self.entity_automation.add_word(k.lower(), (self.entity_vocab[k], k.lower()))
            tmp[k.lower()] = self.entity_vocab[k]
        for k in tmp:
            self.entity_vocab[k] = tmp[k]
        self.entity_automation.make_automaton()

    def _search_entity(self, sentence: str):
        ans = []
        words = set(sentence.split(" "))
        tree = IntervalTree()
        word_index_to_mask = dict()
        ans_index_to_mask = dict()
        mask_val = 100

        for end_index, (insert_order, original_value) in self.entity_automation.iter(sentence):
            start_index = end_index - len(original_value) + 1
            if start_index >= 1 and sentence[start_index-1] != " ":
                continue
            if end_index < len(sentence) - 1 and sentence[end_index+1] != " ":
                continue
            tree.remove_envelop(start_index, end_index)
            should_continue = False
            for item in tree.items():
                if start_index >= item.begin and end_index <= item.end:
                    should_continue = True
                    continue
            if should_continue:
                continue
            if original_value.count(" ") > 0:
                tree.add(Interval(start_index, end_index)) 
            elif original_value in words:
                if len(original_value) > 1:
                    tree.add(Interval(start_index, end_index)) 
        for interval in sorted(tree.items()):
            entity = sentence[interval.begin: interval.end+1]
            entity_index_begin = sentence[0:interval.begin].count(" ")
            entity_index = [entity_index_begin+i for i in range(entity.count(" ")+1)]
            for i in entity_index:
                word_index_to_mask[i] = mask_val
            ans_entity_index_begin = len(ans) 
            ans.append(entity)
            self.tokenizer(sentence[interval.begin: interval.end+1])
            if isinstance(self.entity_vocab[entity], str):
                ans.append("({})".format(self.entity_vocab[entity]))
            ans_entity_index = [i for i in range(ans_entity_index_begin, len(ans))]
            for i in ans_entity_index:
                ans_index_to_mask[i] = mask_val
            mask_val += 1
            #ans.append("$")
        if len(ans) and ans[-1] == "$":
            ans.pop(-1)
        return ans, word_index_to_mask, ans_index_to_mask

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        fields = self.instances[item]
        tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor, position_ids_tensor = self._wrap_data(fields)
        return tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor, position_ids_tensor
    
    def _wrap_data(self, fields):
        sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, subtoken_pos_to_raw_pos, token_type_ids, position_ids = self.parse_line_for_ner(fields=fields)
        tokens_tensor = torch.tensor(tokens_sub_rep, dtype=torch.long)
        #tag_tensor = torch.tensor(coded_ner_, dtype=torch.long).unsqueeze(0)
        tag_tensor = torch.tensor(coded_ner_, dtype=torch.long)
        token_masks_rep = torch.tensor(token_masks_rep)
        token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long)
        position_ids_tensor = torch.tensor(position_ids, dtype=torch.long)
        return tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor, position_ids_tensor

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logging.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in get_ner_reader(data=data):
            if instance_idx < self._min_instances:
                instance_idx += 1
                continue
            if self._max_instances != -1 and instance_idx >= self._max_instances:
                break
            self.instances.append(fields)
            #tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor, = self._wrap_data(fields)
            #self.instances.append((tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor))
            instance_idx += 1
        logging.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def augment_data(self, data, type2type: dict):
        if len(self.type_to_entityset) == 0:
            logging.warning('Please run read_data before running augment_data')
            return
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logging.info('Reading file {} for data augmentation'.format(dataset_name))
        for fields, metadata in get_ner_reader(data=data):
            tokens_, ner_tags = fields[0], fields[-1]
            new_tokens_, new_ner_tags = [], []
            for token, ner_tag in zip(tokens_, ner_tags):
                tag = ner_tag[2:] or "O"
                if tag not in type2type:
                    new_tokens_.append(token)
                    new_ner_tags.append(ner_tag)
                    continue
                rep_tag = type2type[tag]
                if ner_tag.startswith("B-"):
                    entity = random.sample(self.type_to_entityset[rep_tag], 1)[0]
                    new_tokens_.extend(entity.split(" "))
                    for i in range(len(entity.split(" "))):
                        if i == 0:
                            new_ner_tags.append("B-{}".format(tag))
                        else:
                            new_ner_tags.append("I-{}".format(tag))
            tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor, position_ids_tensor = self._wrap_data((new_tokens_, new_ner_tags))
            self.instances.append((tokens_tensor, token_masks_rep, gold_spans_, tag_tensor, subtoken_pos_to_raw_pos, token_type_ids_tensor, position_ids_tensor))
        logging.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def _entity_record(self, fields):
        tokens_, ner_tags = fields[0], fields[-1]
        entity = ""
        tag = ""
        for token, ner_tag in zip(tokens_, ner_tags):
            if ner_tag.startswith("B-"):
                entity = token
                tag = ner_tag[2:]
            elif ner_tag.startswith("I-") or ner_tag.startswith("E-"):
                entity = entity + " " + token
                assert(tag == ner_tag[2:])
            elif ner_tag.startswith("O"):
                if entity:
                    self.label_entities[entity] += 1
                    self.type_to_entityset[tag].add(entity)
                    entity = ""  
            elif ner_tag.startswith("S-"):
                entity = token
                tag = ner_tag[2:]
                self.label_entities[entity] += 1
                self.type_to_entityset[tag].add(entity)
                entity = ""
        if entity:
            self.label_entities[entity] += 1
            self.type_to_entityset[tag].add(entity)

    def parse_line_for_ner(self, fields):
        self._entity_record(fields)
        tokens_, ner_tags = fields[0], fields[-1]
        if len(fields) == 3: # test data, no tag
            ner_tags = ['O' for _ in tokens_]
        sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, subtoken_pos_to_raw_pos, token_type_ids, position_ids = self.parse_tokens_for_ner(tokens_, ner_tags)
        gold_spans_, _ = extract_spans(ner_tags_rep, subtoken_pos_to_raw_pos)
        coded_ner_ = [self.label_to_id[tag] for tag in ner_tags_rep]

        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, subtoken_pos_to_raw_pos, token_type_ids, position_ids

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        sentence_str = ''
        position_ids = []
        tokens_sub_rep, ner_tags_rep = [self.cls_token_id], ['O']
        mask_token = [1]
        token_type_ids = []
        pos_to_single_word = dict()
        subtoken_pos_to_raw_pos = []
        subtoken_pos_to_raw_pos.append(0)
        sentence_str = " ".join(tokens_).lower()
        self.ner_tags.append(ner_tags_rep)
        if self.entity_vocab:
            entity_ans, word_index_to_mask, entity_index_to_mask = self._search_entity(sentence_str)
        else:
            entity_ans, word_index_to_mask, entity_index_to_mask = [], dict(), dict()
        """
                    he    was   named   after   iron   man  [SEP] iron man  (person) [SEP]
        he          1      1      1      1        1     1     1    0    0      0      0
        was         2      1      1      1        1     1     1    0    0      0      0
        named       3      1      1      1        1     1     1    0    0      0      0
        after       4      1      1      1        1     1     1    0    0      0      0
        iron        5      1      1      1        1     1     1    1    1      1      1
        man         5      1      1      1        1     1     1    1    1      1      1
        [SEP]       6      1      1      1        1     1     1    0    0      0      0
        iron        5      
        man         5
        (person)    5
        [SEP]       6
        """ 
        """
            we build the attention matrix via two operations:
            at first, we get the first matrix via NXOR operation between two or two
            secondly, we fix positions representing the first sentence by setting all these values to zero
        """
        sentence_str = ""
        for idx, token in enumerate(tokens_):
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                break
            if sentence_str:
                sentence_str += " " + token.lower()
            else:
                sentence_str = token.lower()
            if idx == 0:
                rep_ = self.tokenizer(token.lower())['input_ids']
            else:
                rep_ = self.tokenizer(" " + token.lower())['input_ids']
            rep_ = rep_[1:-1] #why? the first id is <s>, and the last id is </s>, so we eliminate them
            pos_to_single_word[(len(tokens_sub_rep), len(tokens_sub_rep)+len(rep_))] = token
            subtoken_pos_to_raw_pos.extend([idx+1] * len(rep_))
            tokens_sub_rep.extend(rep_)
            if idx in word_index_to_mask:
                mask_token.extend([word_index_to_mask[idx]] * len(rep_))
            else:
                mask_token.extend([1] * len(rep_))

            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            ner_tag = ner_tags[idx]
            if ner_tag.startswith("B-"):
                self.type_count[ner_tag[2:]] += 1
            tags, masks = _assign_ner_tags(ner_tag, rep_)
            ner_tags_rep.extend(tags)
        self.pos_to_single_word_maps.append(pos_to_single_word)
        tokens_sub_rep.append(self.sep_token_id)
        mask_token.append(1)
        token_type_ids.extend([0] * len(tokens_sub_rep))
        subtoken_pos_to_raw_pos.append(idx+2)
        assert(len(mask_token) == len(tokens_sub_rep)) 
        no_need_to_mask_len = len(tokens_sub_rep)
        #assert(self.tokenizer(sentence_str)["input_ids"] == tokens_sub_rep)
        #assert(len(position_ids) == len(tokens_sub_rep))
        ner_tags_rep.append('O')
        self.ner_tags.append(ner_tags_rep)
        if self.entity_vocab:
            for idx, token in enumerate(entity_ans):
                if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                    break
                if idx == 0:
                    rep_ = self.tokenizer(token.lower())['input_ids']
                else:
                    rep_ = self.tokenizer(" " + token.lower())['input_ids']
                rep_ = rep_[1:-1] #why? the first id is <s>, and the last id is </s>, so we eliminate them
                assert(idx in entity_index_to_mask)
                mask_token.extend([entity_index_to_mask[idx]] * len(rep_))
                tokens_sub_rep.extend(rep_)
            tokens_sub_rep.append(self.sep_token_id)
            mask_token.append(1)

        assert(len(mask_token) == len(tokens_sub_rep))        
        def _nxor(a, b):
            if a == b:
                return 1
            else:
                return 0
        token_masks_rep = [_nxor(i, j) for i, j in itertools.product(mask_token, mask_token)]
        token_masks_rep = np.reshape(np.array(token_masks_rep), newshape=[len(mask_token), len(mask_token)])
        token_masks_rep[:no_need_to_mask_len, :no_need_to_mask_len] = 1
        token_type_ids.extend([1] * (len(tokens_sub_rep) - len(token_type_ids)))

        
        return sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, subtoken_pos_to_raw_pos, token_type_ids, position_ids


if __name__ == "__main__":
    from poros.ner.reader_utils import get_entity_vocab, wnut_iob
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
    entity_vocab = copy.deepcopy(tokenizer.entity_vocab)
    wiki_file = "./data/wiki_def/wiki.pkl.zip"
    wiki_file = "./data/wiki_def/wikigaz.tsv.zip"
    entity_vocab = get_entity_vocab(encoder_model=None, entity_files=[wiki_file])
    conll_reader = CoNLLReader(encoder_model="bert-base-uncased", target_vocab=wnut_iob, entity_vocab=entity_vocab, min_instances=0, max_instances=-1)
    train_file = "./training_data/EN-English/en_train.conll"
    dev_file = "./training_data/EN-English/en_dev.conll"
    test_file = "./training_data/EN-English/en_test.conll"
    conll_reader.read_data(dev_file)
    #conll_reader.augment_data(train_file, {"CORP": "GRP"})
    #conll_reader.augment_data(train_file, {"GRP": "CORP"})
    for batch in conll_reader:
        pass
        #print(batch)