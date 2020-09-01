# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import unittest
import numpy as np
import os
from .dataman import Sample
from .dataman import TrainingInstance
from .dataman import create_attention_mask
from .dataman import PreTrainingDataMan


class SampleTest(unittest.TestCase):

    def setUp(self) -> None:
        self.sample = Sample()

    def test_block_wise_masking(self):
        x = ["[SOS]", "你", "好", "我", "是", "20", "##19", "一", "个", "好",
             "的", "人", "[EOS]", "你", "呢", "你", "在", "干", "嘛", "呢", "[EOS]"]
        for i in range(1000):
            vocab_words = list(set(x).difference(set({"[SOS]", "[EOS]"})))
            M = self.sample.block_wise_masking(x, vocab_words, max_predictions_per_seq=0, mask_ratio=0.15)
            self.assertFalse("[MASK]" in M[0])
            self.assertLessEqual(len(M[4]), 0)
            (output_tokens, output_tokens_positions, masked_lm_positions, masked_lm_labels,
             pseudo_masked_lm_positions, pseudo_masked_lm_labels, pseudo_index, masked_index) = \
                self.sample.block_wise_masking(x, vocab_words, max_predictions_per_seq=4, mask_ratio=0.30)
            print(output_tokens)
            #for i, j in zip(masked_index, masked_lm_positions):
            #    self.assertTrue(output_tokens[i] == '[MASK]' or output_tokens[i] == x[j] or output_tok)
            for i in pseudo_index:
                for j in i:
                    self.assertTrue(output_tokens[j] == '[Pseudo]', "index is {}, tokens is {}".format(j, output_tokens[j]))
            for i, label in zip(masked_lm_positions, masked_lm_labels):
                self.assertTrue(x[i] == label)
            for sublist_index, sublist_labels in zip(pseudo_index, pseudo_masked_lm_labels):
                for i, j in zip(sublist_index, sublist_labels):
                    self.assertTrue(output_tokens[i-len(sublist_index)] == j, "output_tokens[i] is {}, j is {}".format(output_tokens[i], j))


class DatamanTest(unittest.TestCase):

    def test_create_mask_matrix(self):
        tokens = ['x1', 'x2', '[Pesudo]', '[MASK]', 'x3', 'x4', 'x5', '[Pesudo]', '[Pesudo]', '[MASK]', '[MASK]', 'x6', '[PAD]', '[PAD]', '[PAD]']
        output_token_positions = [0, 1, 1, 1, 2, 3, 4, 3, 4, 3, 4, 5, 0, 0, 0]
        input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        segment_ids = [1] * len(output_token_positions)
        is_random_next = False
        pseudo_index = [[7, 8], [2], [], []]
        pseudo_masked_sub_list_len = [2, 1, 0, 0]
        mask_index = [3, 9, 10, 0, 0]
        pseudo_masked_lm_positions = [[3, 4], [1]]
        masked_lm_positions = [2, 5, 6, 0, 0]
        pseudo_masked_lm_labels = [['x2'], ['x4', 'x5']]
        masked_lm_labels = ['x2', 'x4', 'x5']
        instance = TrainingInstance(
            tokens=tokens,
            output_tokens_positions=output_token_positions,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            pseudo_index=pseudo_index,
            pseudo_masked_lm_positions=pseudo_masked_lm_positions,
            masked_lm_positions=masked_lm_positions,
            pseudo_masked_lm_labels=pseudo_masked_lm_labels,
            masked_lm_labels=masked_lm_labels,
            masked_index=mask_index
        )
        flatten_index = [x for _ in instance.pseudo_index for x in _]
        masked_matrix = create_attention_mask(instance.tokens, input_mask, flatten_index, pseudo_masked_sub_list_len)
        expected_matrix = [
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        masked_matrix = np.cast[np.int](masked_matrix)
        print(masked_matrix)
        self.assertListEqual(masked_matrix.tolist(), expected_matrix)

    def test_create_pretrain_data(self):
        vocab_file = os.path.join(os.path.dirname(__file__), "test_data", "vocab.txt")
        input_file = os.path.join(os.path.dirname(__file__), "../bert/sample_text.txt")
        output_file = os.path.join(os.path.dirname(__file__), "test_data", "pretraining_data")
        ptdm = PreTrainingDataMan(vocab_file=vocab_file, max_seq_length=128, max_predictions_per_seq=20)
        ptdm.create_pretraining_data(input_file, output_file)
        dataset = ptdm.read_data_from_tfrecord(output_file, is_training=True, batch_size=1)
        dataset = dataset.repeat(1)
        for data in dataset:
            input_ids = data["input_ids"].numpy()[0]
            input_mask = data["input_mask"].numpy()[0]
            masked_index = data["masked_index"].numpy()[0]
            masked_lm_ids = data["masked_lm_ids"].numpy()[0]
            pseudo_masked_index = data["pseudo_masked_index"].numpy()[0]
            pseudo_masked_lm_ids = data["pseudo_masked_lm_ids"].numpy()[0]
            attention_mask = data["attention_mask"].numpy()[0]
            pseudo_masked_sub_list_len = data["pseudo_masked_sub_list_len"].numpy()[0]
            start_offset = 0
            pseudo_masked_sub_list_len = pseudo_masked_sub_list_len[0:np.count_nonzero(pseudo_masked_sub_list_len)]
            pseudo_masked_index = pseudo_masked_index[0:np.sum(pseudo_masked_sub_list_len)]
            for sublist_len in reversed(pseudo_masked_sub_list_len):
                reversed_pseudo_masked_index = pseudo_masked_index[::-1]
                sublist = reversed_pseudo_masked_index[start_offset:start_offset+sublist_len]
                for i in sublist:
                    for j in sublist:
                        self.assertTrue(attention_mask[i][j] == 1, "i: {}, j: {}, value: {}, {}, {}".format(i, j, attention_mask[i, j],
                                                                                                            pseudo_masked_index, pseudo_masked_sub_list_len))
                        self.assertTrue(attention_mask[i-sublist_len][j-sublist_len] == 1, "i: {}, j: {}, value: {}, {}, {}".format(i, j, attention_mask[i, j],
                                                                                                            pseudo_masked_index, pseudo_masked_sub_list_len))
                start_offset += sublist_len
            for i, value in enumerate(input_mask):
                if value == 0:
                    self.assertTrue(np.sum(attention_mask[i][:]) == 0)
                    self.assertTrue(np.sum(attention_mask[:][i]) == 0)
                    continue


if __name__ == "__main__":
    unittest.main()
