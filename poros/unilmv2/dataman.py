# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import collections
import random
from poros.bert import tokenization
import tensorflow as tf
import numpy as np
import os
from poros.poros_dataset import about_tfrecord
from poros.poros_dataset import about_tensor
import logging


class Sample(object):

    def __init__(self, do_whole_word=False):
        self.do_whole_word = do_whole_word

    def block_wise_masking(self, tokens, vocab_words, max_predictions_per_seq=20, mask_ratio=0.15):
        if not isinstance(tokens, list):
            raise TypeError("input must be list")
        not_include_mask = set()
        for i, ele in enumerate(tokens):
            if ele == "[SOS]" or ele == "[EOS]":
                not_include_mask.add(i)
            if self.do_whole_word and ele.startswith("##"):
                not_include_mask.add(i-1)
                not_include_mask.add(i)

        x_length = len(tokens)
        m_length = min(round(mask_ratio * x_length), max_predictions_per_seq)
        m_set = set()
        pseudo_masked_lm_positions = []
        pseudo_masked_lm_labels = []

        while len(m_set) < m_length:
            p = random.randint(0, x_length-1)
            if random.random() < 0.4:
                l = random.randint(2, 6)
            else:
                l = 1
            sub_list = list(range(p, min(p+l-1+1, x_length)))
            if m_set.intersection(set(sub_list)):
                continue
            if not_include_mask.intersection(set(sub_list)):
                continue
            if len(sub_list) + len(m_set) > m_length:
                continue

            if sub_list.__len__():
                pseudo_masked_lm_positions.append(sub_list)
                pseudo_masked_lm_labels.append([tokens[i] for i in sub_list])
                m_set = m_set.union(set(sub_list))

        sorted_pseudo_masked_lm_positions = sorted(pseudo_masked_lm_positions)
        output_tokens = []
        output_tokens_positions = []
        masked_index = []
        for pos, ele in enumerate(tokens):
            if len(sorted_pseudo_masked_lm_positions):
                pseudo_masked_positions = sorted_pseudo_masked_lm_positions[0]
            else:
                pseudo_masked_positions = []
            output_tokens.append(ele)
            output_tokens_positions.append(pos)
            if pos in pseudo_masked_positions and (pos + 1) not in pseudo_masked_positions:
                for pseudo_position in pseudo_masked_positions:
                    output_tokens.append("[Pseudo]")
                    output_tokens_positions.append(pseudo_position)
                for masked_position in pseudo_masked_positions:
                    if random.random() < 0.8:
                        output_tokens.append("[MASK]")
                    else:
                        if random.random() < 0.5:
                            output_tokens.append(tokens[masked_position])
                        else:
                            output_tokens.append(vocab_words[random.randint(0, len(vocab_words)-1)])
                    masked_index.append(len(output_tokens)-1)
                    output_tokens_positions.append(masked_position)
                sorted_pseudo_masked_lm_positions.pop(0)

        masked_lm_positions = sorted(list(m_set))
        masked_lm_labels = [tokens[i] for i in masked_lm_positions]

        pseudo_index = []
        for ele in pseudo_masked_lm_positions:
            offset = masked_lm_positions.index(ele[0])
            offset = 2 * offset + len(ele)
            sub_pseudo_index = []
            for i in ele:
                sub_pseudo_index.append(i+offset)
            pseudo_index.append(sub_pseudo_index)

        return (output_tokens, output_tokens_positions, masked_lm_positions, masked_lm_labels,
                pseudo_masked_lm_positions, pseudo_masked_lm_labels, pseudo_index, masked_index)


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, output_tokens_positions, segment_ids, masked_lm_positions, masked_lm_labels,
                 pseudo_masked_lm_positions, pseudo_masked_lm_labels, is_random_next, pseudo_index, masked_index):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.output_tokens_positions = output_tokens_positions
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.pseudo_masked_lm_positions = pseudo_masked_lm_positions
        self.pseudo_masked_lm_labels = pseudo_masked_lm_labels
        self.pseudo_index = pseudo_index
        self.masked_index = masked_index

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def create_mask_matrix(instance: TrainingInstance):
    shape = [len(instance.tokens), len(instance.tokens)]
    mask_matrix = np.ones(shape=shape)

    normal_text_can_be_seen = []
    for sub_pseudo_index in reversed(instance.pseudo_index):
        sub_normal_index = [x - len(sub_pseudo_index) for x in sub_pseudo_index]
        normal_text_can_be_seen.extend(sub_normal_index)
        normal_text_can_not_be_seen = list(set(range(shape[0])).difference(set(normal_text_can_be_seen)))
        for _ in sub_normal_index:
            mask_matrix[normal_text_can_not_be_seen, _] = 0

        pseudo_can_be_seen = sub_pseudo_index
        normal_text_can_be_seen.extend(sub_pseudo_index)
        pseudo_can_not_be_seen = list(set(range(shape[0])).difference(set(pseudo_can_be_seen)))
        for _ in sub_pseudo_index:
            mask_matrix[pseudo_can_not_be_seen, _] = 0

    return mask_matrix


def create_attention_mask(input_ids, input_mask, pseudo_masked_index, pseudo_masked_sub_list_len):
    shape = [len(input_ids), len(input_ids)]
    mask_matrix = np.ones(shape=[shape[0], 1], dtype=np.float) * input_mask
    non_zero = np.count_nonzero(input_mask)
    mask_matrix[non_zero:, :] = 0
    mask_matrix[:, non_zero:] = 0
    normal_text_can_be_seen = []
    pseudo_masked_sub_list_len = pseudo_masked_sub_list_len[0:np.count_nonzero(pseudo_masked_sub_list_len)]
    pseudo_masked_index = pseudo_masked_index[0:np.sum(pseudo_masked_sub_list_len)]
    for block_index in pseudo_masked_sub_list_len[::-1]:
        sub_pseudo_index = pseudo_masked_index[-block_index:]
        # Pseudo index correspond to normal index
        sub_normal_index = [x-block_index for x in sub_pseudo_index]
        # normal_text can be seen by sub normal index
        normal_text_can_be_seen.extend(sub_normal_index)
        normal_text_can_not_be_seen = list(set(range(shape[0])).difference(set(normal_text_can_be_seen)))
        for _ in sub_normal_index:
            mask_matrix[normal_text_can_not_be_seen, _] = 0

        pseudo_can_be_seen = sub_pseudo_index
        normal_text_can_be_seen.extend(sub_pseudo_index)
        pseudo_can_not_be_seen = list(set(range(shape[0])).difference(set(pseudo_can_be_seen)))
        for _ in sub_pseudo_index:
            mask_matrix[pseudo_can_not_be_seen, _] = 0
        pseudo_masked_index = pseudo_masked_index[: -block_index]

    return mask_matrix


def add_attention_mask(features, is_training=False):
    if not is_training:
        features["attention_mask"] = None
        return features
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    pseudo_masked_index = features["pseudo_masked_index"]
    pseudo_masked_sub_list_len = features["pseudo_masked_sub_list_len"]
    mask_matrix = tf.numpy_function(create_attention_mask, [input_ids, input_mask, pseudo_masked_index, pseudo_masked_sub_list_len], tf.float64)
    input_ids_shape = about_tensor.get_shape(input_ids, expected_rank=1)
    mask_matrix.set_shape(shape=[input_ids_shape[0], input_ids_shape[0]])
    features["attention_mask"] = mask_matrix

    return features


class PreTrainingDataMan(object):

    def __init__(self, vocab_file, masked_lm_prob=0.15, max_predictions_per_seq=20, do_lower_case=True, max_seq_length=128,
                 dupe_factor=10, short_seq_prob=0.1, do_whole_word=False, random_seed=100):
        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.sample = Sample(do_whole_word=do_whole_word)
        self.dupe_factor = dupe_factor
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.short_seq_prob = short_seq_prob
        self.random_seed = random_seed

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

    def retrieve_id(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def retrieve_tokens(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def write_instance_to_example_files(self, instances, tokenizer, max_seq_length,
                                        max_predictions_per_seq, output_files):
        """Create TF example files from `TrainingInstance`s."""
        writers = []
        for output_file in output_files:
            writers.append(tf.io.TFRecordWriter(output_file))

        writer_index = 0

        total_written = 0
        for (inst_index, instance) in enumerate(instances):
            input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
            segment_ids = list(instance.segment_ids)
            input_mask = [1] * len(instance.tokens)

            origin_input_ids_length = len(input_ids) - 2 * len(instance.masked_lm_positions)
            output_tokens_positions = instance.output_tokens_positions

            assert origin_input_ids_length <= max_seq_length

            #while origin_input_ids_length < max_seq_length:
            while len(input_ids) < (self.max_seq_length + 2*self.max_predictions_per_seq):
                input_ids.append(0)
                input_mask.append(0)
                output_tokens_positions.append(0)
                segment_ids.append(0)
            origin_input_ids_length = len(input_ids) - 2 * self.max_predictions_per_seq

            assert origin_input_ids_length == max_seq_length
            assert len(input_mask) - (2 * self.max_predictions_per_seq) == max_seq_length
            assert len(segment_ids) - (2 * self.max_predictions_per_seq) == max_seq_length
            masked_lm_positions = list(instance.masked_lm_positions)
            masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
            masked_lm_weights = [1.0] * len(masked_lm_ids)
            pseudo_masked_lm_positions = list(instance.pseudo_masked_lm_positions)
            pseudo_masked_lm_ids = [tokenizer.convert_tokens_to_ids(_) for _ in instance.pseudo_masked_lm_labels ]
            pseudo_masked_index = instance.pseudo_index
            masked_index = instance.masked_index
            pseudo_masked_sub_list_len = [len(sub_list) for sub_list in pseudo_masked_lm_positions]

            while len(masked_lm_positions) < max_predictions_per_seq:
                masked_lm_positions.append(0)
                masked_lm_ids.append(0)
                masked_lm_weights.append(0.0)
                pseudo_masked_lm_positions.append([0])
                pseudo_masked_index.append([0])
                masked_index.append(0)
                pseudo_masked_lm_ids.append([0])

            while len(pseudo_masked_sub_list_len) < max_predictions_per_seq:
                pseudo_masked_sub_list_len.append(0)

            next_sentence_label = 1 if instance.is_random_next else 0

            features = collections.OrderedDict()
            features["input_ids"] = about_tfrecord.create_int_feature(input_ids)
            # convert array to a byte_feature
            features["input_mask"] = about_tfrecord.create_int_feature(input_mask)

            flatten_pseudo_masked_lm_positions = [_ for sub_list in pseudo_masked_lm_positions for _ in sub_list]
            #features["pseudo_masked_lm_positions"] = about_tfrecord.create_int_feature(flatten_pseudo_masked_lm_positions)
            features["pseudo_masked_sub_list_len"] = about_tfrecord.create_int_feature(pseudo_masked_sub_list_len)
            flatten_pseudo_masked_index = [_ for sub_list in pseudo_masked_index for _ in sub_list]
            features["pseudo_masked_index"] = about_tfrecord.create_int_feature(flatten_pseudo_masked_index)
            features["masked_index"] = about_tfrecord.create_int_feature(masked_index)
            flatten_pseudo_masked_lm_ids = [_ for sub_list in pseudo_masked_lm_ids for _ in sub_list]
            features["pseudo_masked_lm_ids"] = about_tfrecord.create_int_feature(flatten_pseudo_masked_lm_ids)
            features["output_tokens_positions"] = about_tfrecord.create_int_feature(output_tokens_positions)
            features["segment_ids"] = about_tfrecord.create_int_feature(segment_ids)
            #features["masked_lm_positions"] = about_tfrecord.create_int_feature(masked_lm_positions)
            features["masked_lm_ids"] = about_tfrecord.create_int_feature(masked_lm_ids)
            features["masked_lm_weights"] = about_tfrecord.create_float_feature(masked_lm_weights)
            features["next_sentence_labels"] = about_tfrecord.create_int_feature([next_sentence_label])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))

            writers[writer_index].write(tf_example.SerializeToString())
            writer_index = (writer_index + 1) % len(writers)

            total_written += 1

            if inst_index < 20:
                tf.get_logger().info("*** Example ***")
                tf.get_logger().info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in instance.tokens]))

                for feature_name in features.keys():
                    feature = features[feature_name]
                    values = []
                    if feature.int64_list.value:
                        values = feature.int64_list.value
                    elif feature.float_list.value:
                        values = feature.float_list.value
                    tf.get_logger().info(
                        "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

        for writer in writers:
            writer.close()
        tf.get_logger().info("Wrote %d total instances", total_written)

    def create_training_instances(self, input_files, tokenizer, max_seq_length,
                                  dupe_factor, short_seq_prob, masked_lm_prob,
                                  max_predictions_per_seq, rng):
        """Create `TrainingInstance`s from raw text."""
        all_documents = [[]]

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        for input_file in input_files:
            with tf.io.gfile.GFile(input_file, "r") as reader:
                while True:
                    line = tokenization.convert_to_unicode(reader.readline())
                    if not line:
                        break
                    line = line.strip()

                    # Empty lines are used as document delimiters
                    if not line:
                        all_documents.append([])
                    tokens = tokenizer.tokenize(line)
                    if tokens:
                        all_documents[-1].append(tokens)

        # Remove empty documents
        all_documents = [x for x in all_documents if x]
        rng.shuffle(all_documents)

        vocab_words = list(tokenizer.vocab.keys())
        instances = []
        for _ in range(dupe_factor):
            for document_index in range(len(all_documents)):
                instances.extend(
                    self.create_instances_from_document(
                        all_documents, document_index, max_seq_length, short_seq_prob,
                        masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

        rng.shuffle(instances)
        return instances

    def create_instances_from_document(self,
            all_documents, document_index, max_seq_length, short_seq_prob,
            masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
        """Creates `TrainingInstance`s for a single document."""
        document = all_documents[document_index]

        # Account for [SOS], [EOS], [EOS]
        max_num_tokens = max_seq_length - 3

        # We *usually* want to fill up the entire sequence since we are padding
        # to `max_seq_length` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `max_seq_length` is a hard limit.
        target_seq_length = max_num_tokens
        if rng.random() < short_seq_prob:
            target_seq_length = rng.randint(2, max_num_tokens)

        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input.
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = rng.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    # Random next
                    is_random_next = False
                    if len(current_chunk) == 1 or rng.random() < 0.5:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(20):
                            random_document_index = rng.randint(0, len(all_documents) - 1)
                            # faq documents maybe has the same answer, so we compare the content with each other
                            if all_documents[random_document_index] != all_documents[document_index]:
                                break
                            if random_document_index != document_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = rng.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                    self.truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    tokens = []
                    segment_ids = []
                    tokens.append("[SOS]")
                    segment_ids.append(0)
                    for token in tokens_a:
                        tokens.append(token)

                    tokens.append("[EOS]")

                    for token in tokens_b:
                        tokens.append(token)
                    tokens.append("[EOS]")

                    (tokens, output_tokens_positions, masked_lm_positions, masked_lm_labels, pseudo_masked_lm_positions, pseudo_masked_lm_labels, pseudo_index, masked_index) =\
                        self.sample.block_wise_masking(tokens, vocab_words, max_predictions_per_seq, masked_lm_prob)
                    segment_id = 0
                    segment_ids = []
                    for token in tokens:
                        segment_ids.append(segment_id)
                        if token == "[SOS]":
                            segment_id = 0
                        if token == "[EOS]":
                            segment_id = 1

                    instance = TrainingInstance(
                        tokens=tokens,
                        segment_ids=segment_ids,
                        is_random_next=is_random_next,
                        output_tokens_positions=output_tokens_positions,
                        masked_lm_positions=masked_lm_positions,
                        masked_lm_labels=masked_lm_labels,
                        pseudo_masked_lm_positions=pseudo_masked_lm_positions,
                        pseudo_masked_lm_labels=pseudo_masked_lm_labels,
                        pseudo_index=pseudo_index,
                        masked_index=masked_index
                    )
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1

        return instances

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens, rng):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break

            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if rng.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def create_pretraining_data(self, input_file, output_file):
        tf.get_logger().setLevel(logging.INFO)

        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

        input_files = []
        for input_pattern in input_file.split(","):
            input_files.extend(tf.io.gfile.glob(input_pattern))

        tf.get_logger().info("*** Reading from input files ***")
        for input_file in input_files:
            tf.get_logger().info("  %s", input_file)

        rng = random.Random(self.random_seed)
        instances = self.create_training_instances(
            input_files, tokenizer, self.max_seq_length, self.dupe_factor,
            self.short_seq_prob, self.masked_lm_prob, self.max_predictions_per_seq,
            rng)

        output_files = output_file.split(",")
        tf.get_logger().info("*** Writing to output files ***")
        for output_file in output_files:
            tf.get_logger().info("  %s", output_file)

        self.write_instance_to_example_files(instances, tokenizer, self.max_seq_length,
                                        self.max_predictions_per_seq, output_files)

    def read_data_from_tfrecord(self, input_files, is_training, batch_size=128, num_cpu_threads=0):

        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([self.max_seq_length+self.max_predictions_per_seq*2], tf.int64),
            #"pseudo_masked_lm_positions": tf.io.FixedLenFeature([self.max_predictions_per_seq], tf.int64),
            "input_mask": tf.io.FixedLenFeature([self.max_seq_length+self.max_predictions_per_seq*2], tf.int64),
            "pseudo_masked_index": tf.io.FixedLenFeature([self.max_predictions_per_seq], tf.int64),
            "masked_index": tf.io.FixedLenFeature([self.max_predictions_per_seq], tf.int64),
            "pseudo_masked_sub_list_len": tf.io.FixedLenFeature([self.max_predictions_per_seq], tf.int64),
            "pseudo_masked_lm_ids": tf.io.FixedLenFeature([self.max_predictions_per_seq], tf.int64),
            "output_tokens_positions": tf.io.FixedLenFeature([self.max_seq_length+self.max_predictions_per_seq*2], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([self.max_seq_length+self.max_predictions_per_seq*2], tf.int64),
            #"masked_lm_positions": tf.io.FixedLenFeature([self.max_predictions_per_seq], tf.int64),
            "masked_lm_ids": tf.io.FixedLenFeature([self.max_predictions_per_seq], tf.int64),
            "masked_lm_weights": tf.io.FixedLenFeature([self.max_predictions_per_seq], tf.float32),
            "next_sentence_labels": tf.io.FixedLenFeature([], tf.int64)
        }

        if not isinstance(input_files, list):
            input_files = [input_files]
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(input_files)
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            if num_cpu_threads == 0:
                cycle_length = min(os.cpu_count(), len(input_files))
            else:
                cycle_length = num_cpu_threads

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(tf.data.experimental.parallel_interleave(map_func=tf.data.TFRecordDataset,
                                                                 cycle_length=cycle_length,
                                                                 sloppy=is_training
                                                                 ))
            d = d.shuffle(buffer_size=1000)
        else:
            d = tf.data.TFRecordDataset(input_files)
        d = d.map(lambda record: about_tfrecord.parse_example(record, name_to_features))
        d = d.map(lambda x: add_attention_mask(x, is_training))

        d = d.batch(batch_size=batch_size, drop_remainder=True)

        return d


if __name__ == "__main__":
    input_file = "../bert/sample_text.txt"
    output_file = "./pretraining_data"
    vocab_file = "./test_data/vocab.txt"
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    ptdm = PreTrainingDataMan(vocab_file=vocab_file, max_seq_length=128)
    if not os.path.exists(output_file):
        ptdm.create_pretraining_data(input_file, output_file)
    dataset = ptdm.read_data_from_tfrecord(output_file, is_training=True, batch_size=1)
    for data in dataset:
        ids = data["input_ids"].numpy()
        print(tokenizer.convert_ids_to_tokens(ids[0]))
