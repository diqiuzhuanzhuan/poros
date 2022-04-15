# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

import gzip
import itertools
from typing import List
import zipfile
import os
import pickle

conll_iob = {'B-ORG': 0, 'I-ORG': 1, 'B-MISC': 2, 'I-MISC': 3, 'B-LOC': 4, 'I-LOC': 5, 'B-PER': 6, 'I-PER': 7, 'O': 8}
wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7, 'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}
luke_iob = {'CORP': 0, 'CW': 1, 'GRP': 2, 'LOC':3, 'PER': 4, 'PROD': 5, 'O': 6}


def get_ner_reader(data):
    fin = gzip.open(data, 'rt') if data.endswith('.gz') else open(data, 'rt')
    for is_divider, lines in itertools.groupby(fin, _is_divider):
        if is_divider:
            continue
        lines = [line.strip().replace('\u200d', '').replace('\u200c', '') for line in lines]

        metadata = lines[0].strip() if lines[0].strip().startswith('# id') else None
        fields = [line.split() for line in lines if not line.startswith('# id')]
        fields = [list(field) for field in zip(*fields)]


        yield fields, metadata


def _assign_ner_tags(ner_tag, rep_):
    ner_tags_rep = []
    token_masks = []

    sub_token_len = len(rep_)
    token_masks.extend([True] * sub_token_len)
    if ner_tag[0] == 'B':
        in_tag = 'I' + ner_tag[1:]

        ner_tags_rep.append(ner_tag)
        ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
    else:
        ner_tags_rep.extend([ner_tag] * sub_token_len)
    return ner_tags_rep, token_masks


def extract_spans(tags, subtoken_pos_to_raw_pos):
    cur_tag = None
    cur_start = None
    gold_spans = {}

    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        if _cur_start is None:
            return _gold_spans
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag  # inclusive start & end, accord with conll-coref settings
        return _gold_spans

    # iterate over the tags
    actual_tags = []
    last_pos = None
    for _id, (nt, pos) in enumerate(zip(tags, subtoken_pos_to_raw_pos)):
        indicator = nt[0]
        """
        if _id == 0 or pos != subtoken_pos_to_raw_pos[_id-1]:
            new_word = True
        else:
            new_word = False
        if cur_tag and new_word:
            _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt
        elif new_word:
            cur_tag = nt
            cur_start = _id
        else:
            pass
        """
        if indicator == 'B':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        elif indicator == 'I':
            # do nothing
            pass
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = 'O'
            cur_start = _id
            pass
        if pos == last_pos:
            pass
        else:
            actual_tags.append(nt)
        last_pos = pos
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    assert(len(actual_tags) == len(set(subtoken_pos_to_raw_pos)))
    actual_tags.pop(0) # CLS
    actual_tags.pop(-1) # SEP
    return gold_spans, actual_tags


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True

    first_token = line.split()[0]
    if first_token == "-DOCSTART-":# or line.startswith('# id'):  # pylint: disable=simplifiable-if-statement
        return True

    return False


def get_tags(tokens, tags, tokenizer=None, start_token_pattern='▁'):
    token_results, tag_results = [], []
    index = 0
    token_word = []
    tokens = tokenizer.convert_ids_to_tokens(tokens)
    for token, tag in zip(tokens, tags):
        if token == tokenizer.pad_token:
            # index += 1
            continue

        if index == 0:
            tag_results.append(tag)

        elif token.startswith(start_token_pattern) and token != '▁́':
            tag_results.append(tag)

            if tokenizer is not None:
                token_results.append(''.join(token_word).replace(start_token_pattern, ''))
            token_word.clear()

        token_word.append(token)

        index += 1
    token_results.append(''.join(token_word).replace(start_token_pattern, ''))

    return token_results, tag_results

def get_entity_vocab(encoder_model="studio-ousia/luke-base", conll_files: List[str]=[], entity_files: List[str]=[]):
    from transformers import LukeTokenizer
    import copy
    if encoder_model:
        tokenizer = LukeTokenizer.from_pretrained(encoder_model)
        entity_vocab = copy.deepcopy(tokenizer.entity_vocab)
    else:
        entity_vocab = dict()
    def _get_entity(fields, entity_set):
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
                    entity_set.add(entity)
                    entity = ""  
            elif ner_tag.startswith("S-"):
                entity = token
                tag = ner_tag[2:]
                entity_set.add(entity)
                entity = ""
        if entity:
            entity_set.add(entity)

    for file in conll_files:
        entity_set = set()
        for fields, _ in get_ner_reader(file):
            _get_entity(fields, entity_set)
        
        for entity in entity_set:
            if entity not in entity_vocab:
                entity_vocab[entity] = len(entity_vocab)

    for file in entity_files:
        if file.endswith(".zip"):
            with zipfile.ZipFile(file) as myzip:
                with myzip.open(os.path.basename(file.strip(".zip"))) as f:
                    if "pkl" in file:
                        wiki_data = pickle.load(f)
                        for entity in wiki_data:
                            if len(wiki_data[entity]) == 0:
                                entity_vocab[entity.lower()] = len(entity_vocab)
                            else:
                                entity_vocab[entity.lower()] = wiki_data[entity][0]
                    elif "tsv" in file:
                        _map_type_to_human_readable_words = {
                            "LOC": "location",
                            "CORP": "corperation",
                            "GRP": "group",
                            "PER": "person",
                            "CW": "creative work",
                            "PROD": "product",
                        }
                        for line in f:
                            fields = line.decode("utf-8").strip("\n").strip("\r").split("\t")
                            if len(fields) != 4:
                                continue
                            entity, entity_type = fields[3].lower(), _map_type_to_human_readable_words[fields[1]]
                            if entity in entity_vocab:
                                if not isinstance(entity_vocab[entity.lower()], str):
                                    entity_vocab[entity.lower()] = entity_type

                                if entity_type not in entity_vocab[entity]:
                                    entity_vocab[entity.lower()] = entity_vocab[entity.lower()] + "|" + entity_type
                            else:
                                entity_vocab[entity.lower()] = entity_type
                            
                    else:    
                        for entity in f:
                            entity = entity.strip('\r').strip('\n')
                            entity_vocab[entity] = len(entity_vocab)
                            if " (" in entity: 
                                entity = entity.split(" (")[0]
                                entity_vocab[entity] = len(entity_vocab)
                        
        else:
            with open(file, "r") as f:
                for entity in f:
                    entity = entity.strip('\r').strip('\n')
                    if not entity:
                        continue
                    if entity not in entity_vocab:
                        entity_vocab[entity] = len(entity_vocab)
                    if " (" in entity: 
                        entity = entity.split(" (")[0]
                        if entity not in entity_vocab:
                            entity_vocab[entity] = len(entity_vocab)
    
    return entity_vocab