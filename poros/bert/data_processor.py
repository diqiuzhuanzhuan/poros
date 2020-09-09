# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import tensorflow as tf
import csv
import os
import collections
from poros.poros_dataset import about_tfrecord
from poros.bert import tokenization

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class SiameseFeatures(object):

    def __init__(self,
                 input_ids_a,
                 input_mask_a,
                 input_ids_b,
                 input_mask_b,
                 label_id):
        self.input_ids_a = input_ids_a
        self.input_mask_a = input_mask_a
        self.input_ids_b = input_ids_b
        self.input_mask_b = input_mask_b
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class XnliProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(
            os.path.join(data_dir, "multinli",
                         "multinli.train.%s.tsv" % self.language))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            if label == tokenization.convert_to_unicode("contradictory"):
                label = tokenization.convert_to_unicode("contradiction")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            language = tokenization.convert_to_unicode(line[0])
            if language != tokenization.convert_to_unicode(self.language):
                continue
            text_a = tokenization.convert_to_unicode(line[6])
            text_b = tokenization.convert_to_unicode(line[7])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            if set_type == "test":
                label = "contradiction"
            else:
                label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the Cola data set (GLUE version)"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[3])
                label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class SiameseProcessor(DataProcessor):

    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.name_to_features = {
            "input_ids_a": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask_a": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_ids_b": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask_b": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "label_id": tf.io.FixedLenFeature([1], tf.int64)
        }

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_examples_not_from_file(self, lines, set_type):
        return self._create_examples(lines, set_type)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def convert_examples_to_features(self, examples):
        res = []
        for e_index, example in enumerate(examples):
            feature = self._convert_single_example(e_index, example)
            res.append(feature)
        return res

    def write_features_into_tfrecord(self, features, filename):
        writer = tf.io.TFRecordWriter(filename)
        for ele in features:
            features = collections.OrderedDict()
            features["input_ids_a"] = about_tfrecord._int64_feature(ele.input_ids_a)
            features["input_ids_b"] = about_tfrecord._int64_feature(ele.input_ids_b)
            features["label_id"] = about_tfrecord._int64_feature([ele.label_id])
            writer.write(about_tfrecord.serialize_example(features))

        writer.close()

    def read_features_from_tfrecord(self, filename):
        d = tf.data.TFRecordDataset(filename)
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: about_tfrecord.parse_example(record, self.name_to_features))
        )
        return d

    @staticmethod
    def decode_record(self, record):
        return {"input_ids_a": record[0], "input_ids_b": record[1], "label_id": [record[2]]}

    def _convert_single_example(self, ex_index, example):
        label_map = {}
        for (i, label) in enumerate(self.get_labels()):
            label_map[label] = i

        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = self.tokenizer.tokenize(example.text_b)
        if tokens_a:
            # Account for [CLS] and [SEP] with "- 2"
            tokens_a = tokens_a[0:(self.max_seq_length - 2)]

        if tokens_b:
            # Account for [CLS] and [SEP] with "- 2"
            tokens_b = tokens_b[0:(self.max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")
        input_ids_a = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask_a = [1] * len(input_ids_a)

        tokens = []
        tokens.append("[CLS]")
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
            tokens.append("[SEP]")

        input_ids_b = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask_b = [1] * len(input_ids_b)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.

        # Zero-pad up to the sequence length.
        while len(input_ids_a) < self.max_seq_length:
            input_ids_a.append(0)
            input_mask_a.append(0)

        while len(input_ids_b) < self.max_seq_length:
            input_ids_b.append(0)
            input_mask_b.append(0)

        assert len(input_ids_a) == self.max_seq_length
        assert len(input_ids_b) == self.max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            tf.get_logger().info("*** Example ***")
            tf.get_logger().info("guid: %s" % (example.guid))
            tf.get_logger().info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.get_logger().info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
            tf.get_logger().info("input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
            tf.get_logger().info("input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
            tf.get_logger().info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
            tf.get_logger().info("label: %s (id = %d)" % (example.label, label_id))

        feature = SiameseFeatures(
            input_ids_a=input_ids_a,
            input_mask_a=input_mask_a,
            input_ids_b=input_ids_b,
            input_mask_b=input_mask_b,
            label_id=label_id)
        return feature

    def convert_features(self, feature: SiameseFeatures):

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = {
            "input_ids_a": create_int_feature(feature.input_ids_a),
            "input_mask_a": create_int_feature(feature.input_mask_a),
            "input_ids_b": create_int_feature(feature.input_ids_b),
            "input_mask_b": create_int_feature(feature.input_mask_b),
            "label_id": create_int_feature([feature.label_id])

        }
        return tf.train.Features(feature=features)
