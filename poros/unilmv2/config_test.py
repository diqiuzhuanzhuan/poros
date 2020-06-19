# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import unittest
import os
from poros.unilmv2 import Unilmv2Config


class Unilmv2ConfigTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

    def test_from_json_file(self):
        json_file = os.path.join(self.test_data_path, "bert_config.json")
        unilmv2_config = Unilmv2Config.from_json_file(json_file)
        expected_dict = {
            "attention_probs_dropout_prob": 0.9,
            "directionality": "bidi",
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.9,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pooler_fc_size": 768,
            "pooler_num_attention_heads": 12,
            "pooler_num_fc_layers": 3,
            "pooler_size_per_head": 128,
            "pooler_type": "first_token_transform",
            "type_vocab_size": 2,
            "vocab_size": 21132
        }

        self.assertDictEqual(unilmv2_config.to_dict(), expected_dict)
