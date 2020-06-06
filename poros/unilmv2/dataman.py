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
from poros.poros_dataset import about_tfrecord


class Sample(object):

    def __init__(self, do_whole_word=False):
        self.do_whole_word = do_whole_word

    def block_wise_masking(self, tokens, max_predictions_per_seq=20, mask_ratio=0.15):
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
        pesudo_masked_lm_labels = []

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
                pesudo_masked_lm_labels.append([tokens[i] for i in sub_list])
                m_set = m_set.union(set(sub_list))

        sorted_pseudo_masked_lm_positions = sorted(pseudo_masked_lm_positions)
        output_tokens = []
        output_tokens_positions = []
        pseudo_index = []
        mask_index = []
        for pos, ele in enumerate(tokens):
            if len(sorted_pseudo_masked_lm_positions):
                pseudo_masked_positions = sorted_pseudo_masked_lm_positions[0]
            else:
                pseudo_masked_positions = []
            output_tokens.append(ele)
            output_tokens_positions.append(pos)
            if pos in pseudo_masked_positions and (pos + 1) not in pseudo_masked_positions:
                sub_pseudo_index = []
                for pseudo_position in pseudo_masked_positions:
                    output_tokens.append("[Pseudo]")
                    sub_pseudo_index.append(len(output_tokens) -1 )
                    output_tokens_positions.append(pseudo_position)
                for masked_position in pseudo_masked_positions:
                    output_tokens.append("[MASK]")
                    mask_index.append(len(output_tokens)-1)
                    output_tokens_positions.append(masked_position)
                sorted_pseudo_masked_lm_positions.pop(0)
                pseudo_index.append(sub_pseudo_index)

        masked_lm_positions = sorted(list(m_set))
        masked_lm_labels = [tokens[i] for i in masked_lm_positions]

        return (output_tokens, output_tokens_positions, masked_lm_positions, masked_lm_labels,
                pseudo_masked_lm_positions, pesudo_masked_lm_labels, pseudo_index, mask_index)


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, output_tokens_positions, segment_ids, masked_lm_positions, masked_lm_labels,
                 pseudo_masked_lm_positions, pseudo_masked_lm_labels, is_random_next, pseudo_index, mask_index):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.output_tokens_positions = output_tokens_positions
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.pseudo_masked_lm_positions = pseudo_masked_lm_positions
        self.pseudo_masked_lm_labels = pseudo_masked_lm_labels
        self.pseudo_index = pseudo_index
        self.mask_index = mask_index

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
            input_mask = create_mask_matrix(instance)
            segment_ids = list(instance.segment_ids)

            origin_input_ids_length = len(input_ids) - 2 * len(instance.masked_lm_positions)

            assert origin_input_ids_length <= max_seq_length

            while origin_input_ids_length < max_seq_length:
                input_ids.append(0)
                input_mask = np.pad(input_mask, [(0, 1), (0, 1)])
                segment_ids.append(0)
                origin_input_ids_length = len(input_ids) - 2 * len(instance.masked_lm_positions)

            assert origin_input_ids_length == max_seq_length
            assert len(input_mask) - (2 * len(instance.masked_lm_positions)) == max_seq_length
            assert len(segment_ids) - (2 * len(instance.masked_lm_positions)) == max_seq_length

            output_tokens_positions = instance.output_tokens_positions
            masked_lm_positions = list(instance.masked_lm_positions)
            masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
            masked_lm_weights = [1.0] * len(masked_lm_ids)
            pseudo_masked_lm_positions = list(instance.pseudo_masked_lm_positions)
            pseudo_masked_lm_ids = [tokenizer.convert_tokens_to_ids(_) for _ in instance.pseudo_masked_lm_labels ]

            while len(masked_lm_positions) < max_predictions_per_seq:
                masked_lm_positions.append(0)
                masked_lm_ids.append(0)
                masked_lm_weights.append(0.0)

            next_sentence_label = 1 if instance.is_random_next else 0

            features = collections.OrderedDict()
            features["input_ids"] = about_tfrecord.create_int_feature(input_ids)
            # convert array to a byte_feature
            features["input_mask"] = about_tfrecord.create_bytes_feature(about_tfrecord.serialize_array(input_mask))
            flatten_pseudo_masked_lm_positions = [_ for sub_list in pseudo_masked_lm_positions for _ in sub_list]
            features["pseudo_masked_lm_positions"] = about_tfrecord.create_int_feature(flatten_pseudo_masked_lm_positions)
            flatten_pseudo_masked_lm_ids = [_ for sub_list in pseudo_masked_lm_ids for _ in sub_list]
            features["pseudo_masked_lm_ids"] = about_tfrecord.create_int_feature(flatten_pseudo_masked_lm_ids)
            features["output_tokens_positions"] = about_tfrecord.create_int_feature(output_tokens_positions)
            features["segment_ids"] = about_tfrecord.create_int_feature(segment_ids)
            features["masked_lm_positions"] = about_tfrecord.create_int_feature(masked_lm_positions)
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

                    (tokens, output_tokens_positions, masked_lm_positions, masked_lm_labels, pseudo_masked_lm_positions, pseudo_masked_lm_labels, pseudo_index, mask_index) =\
                        self.sample.block_wise_masking(tokens, max_predictions_per_seq, masked_lm_prob)
                    segment_id = 0
                    segment_ids = []
                    for token in tokens:
                        segment_ids.append(segment_id)
                        if token == "[SOS]":
                            segment_id = 0
                        if token == "[EOS]":
                            segment_id = 1

                    """
                    (tokens, masked_lm_positions,
                     masked_lm_labels) = self.create_masked_lm_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                    """

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
                        mask_index=mask_index
                    )
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1

        return instances

    def create_masked_lm_predictions(self, tokens, masked_lm_prob,
                                     max_predictions_per_seq, vocab_words, rng, do_whole_word_mask=False):
        """Creates the predictions for the masked LM objective."""

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[SOS]" or token == "[EOS]":
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word indexes.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
            if (do_whole_word_mask and len(cand_indexes) >= 1 and
                    token.startswith("##")):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        rng.shuffle(cand_indexes)

        output_tokens = list(tokens)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

                masked_token = None

                # 80% of the time, replace with [MASK]
                if rng.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

                output_tokens[index] = masked_token

                masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)

        return (output_tokens, masked_lm_positions, masked_lm_labels)

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

    def main(self, input_file, output_file):
        import logging
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


if __name__ == "__main__":

    input_file = "../bert/sample_text.txt"
    output_file = "./pretraining_data"
    vocab_file = "../bert_model/data/chinese_L-12_H-768_A-12/vocab.txt"
    ptdm = PreTrainingDataMan(vocab_file=vocab_file, max_seq_length=256)
    ptdm.main(input_file, output_file)
