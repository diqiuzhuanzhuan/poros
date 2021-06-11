# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from datasets import load_dataset
import tensorflow as tf
from transformers import BertTokenizer
import functools


class SnliDataMan(object):
    def __init__(self, pretrain_name='bert-base-uncased'):
        self.dataset = dict()
        self.dataset["train"] = load_dataset('snli', split='train')
        self.dataset["validation"] = load_dataset('snli', split='validation')
        self.dataset["test"] = load_dataset('snli', split='test')
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrain_name)

    @classmethod
    def _build_data(cls, data):
        return

    def gen(self, data_type="train"):
        try:
            data = self.dataset[data_type]
        except KeyError as e:
            raise ValueError("data_type muse be 'train' or 'validation' or 'test'")

        for ele in data:
            inputs = dict()
            input_a = self.bert_tokenizer(ele['premise'])
            input_b = self.bert_tokenizer(ele['hypothesis'])
            inputs["token_type_ids"] = self.bert_tokenizer.create_token_type_ids_from_sequences(
                input_a["input_ids"], input_b["input_ids"]
            )
            length = len(inputs['token_type_ids'])
            inputs["token_type_ids"] += [0] * (self.bert_tokenizer.model_max_length - length)
            inputs["input_ids"] = self.bert_tokenizer.build_inputs_with_special_tokens(
                input_a["input_ids"], input_b["input_ids"]
            ) + self.bert_tokenizer.convert_tokens_to_ids(['[PAD]']) * (self.bert_tokenizer.model_max_length - length)
            inputs["attention_mask"] = [1] * length + [0] * (self.bert_tokenizer.model_max_length - length)
            inputs["label_ids"] = ele.get('label', 0)
            yield inputs, {}

    def batch(self, data_type='train', batch_size=32, repeat=None, shuffle=1000):
        data = functools.partial(self.gen, data_type)
        tf_data = tf.data.Dataset.from_generator(data, output_types=(
            {
                'token_type_ids': tf.int32,
                'input_ids': tf.int32,
                'attention_mask': tf.int32,
                'label_ids': tf.int32
            },
            {}), output_shapes=(
            {
                'token_type_ids': tf.TensorShape((self.bert_tokenizer.model_max_length,)),
                'input_ids': tf.TensorShape((self.bert_tokenizer.model_max_length,)),
                'attention_mask': tf.TensorShape((self.bert_tokenizer.model_max_length,)),
                'label_ids': tf.TensorShape(())
            },
            {},
        ))
        tf_data = tf_data.batch(batch_size=batch_size).repeat(repeat).shuffle(shuffle)
        return tf_data


if __name__ == "__main__":
    snli_dataman = SnliDataMan()
    for i in snli_dataman.batch(data_type='train', batch_size=32):
        print(i[0]['input_ids'].shape)