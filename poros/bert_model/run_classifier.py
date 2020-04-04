# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import collections
import csv
import os
from poros.bert_model import modeling
from poros.bert_model import optimization
from poros.bert_model import tokenization
from poros.poros_loss import about_loss
import tensorflow as tf


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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
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
    def _read_csv(cls, input_file, quotechar="\""):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class JustClassifierDataProcessor(DataProcessor):
    def __init__(self):
        self.labels = None

    def set_labels(self, labels):
        self.labels = labels

    def get_train_examples(self, file):
        """See base class."""
        return self._create_examples(
            self._read_csv(file), "train")

    def get_dev_examples(self, file):
        """See base class."""
        return self._create_examples(
            self._read_csv(file), "dev")

    def get_test_examples(self, file):
        """See base class."""
        return self._create_examples(
            self._read_csv(file), "test")

    def get_test_examples_not_from_file(self, lines):
        return self._create_examples(lines, set_type="test", skip_header=0)

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, lines, set_type, skip_header=0):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if skip_header > i:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = self.labels[0]
            else:
                text_a = tokenization.convert_to_unicode(line[1])
                label = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

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
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        import logging
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


def convert_feature_to_tf_example(feature):
    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)
        tf_example = convert_feature_to_tf_example(feature)

        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, batch_size=8):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        #batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        arg_max = tf.argmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        #per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = about_loss.focal_loss(log_probs, one_hot_labels)

        return (loss, logits, probabilities, arg_max, output_layer)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits, probabilities, arg_max, output_layer) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        _, average_loss = tf.metrics.mean(total_loss, name="average_loss")

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        serving_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        predictions = {
            "classes": arg_max,
            "prob": probabilities,
            "output_layer": output_layer
        }
        export_outputs = {serving_key: tf.estimator.export.PredictOutput(predictions)}

        if mode == tf.estimator.ModeKeys.TRAIN:
            hook = tf.train.LoggingTensorHook(tensors={"average_loss": "average_loss/update_op"}, every_n_iter=100)
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=tf.group([train_op, average_loss]),
                scaffold=scaffold_fn,
                training_hooks=[hook]
            )
            """
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
            """
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                recall = tf.metrics.recall(labels=label_ids, predictions=predictions)
                loss = tf.metrics.mean(per_example_loss)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    "recall": recall
                }
            eval_metrics = metric_fn(total_loss, label_ids, logits)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                scaffold=scaffold_fn,
                export_outputs=export_outputs
            )
            """
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=arg_max, scaffold_fn=scaffold_fn, export_outputs=export_outputs)
            """
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder, batch_size=8):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        #batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def queue_input_fn_builder(features, seq_length, is_training, drop_remainder):
    pass


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


class SimpleClassifierModel(object):

    def __init__(self,
                 bert_config_file,
                 vocab_file,
                 output_dir,
                 max_seq_length,
                 train_file,
                 dev_file,
                 init_checkpoint,
                 train_batch_size=32,
                 eval_batch_size=32,
                 predict_batch_size=64,
                 is_train=False,
                 label_list=["0", "1"],
                 learning_rate=5e-5,
                 warmup_proportion=0.1,
                 save_checkpoints_steps=1000,
                 iterations_per_loop=1000,
                 num_train_epochs=5):

        tf.set_random_seed(100)
        tf.logging.set_verbosity(tf.logging.INFO)
        self.is_train = is_train
        self.bert_config_file = bert_config_file
        self.vocab_file = vocab_file
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.warmup_proportion = warmup_proportion
        self.train_file = train_file
        self.dev_file = dev_file
        self.init_checkpoint = init_checkpoint
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.predict_batch_size = predict_batch_size
        self.save_checkpoints_steps = save_checkpoints_steps
        self.iterations_per_loop = iterations_per_loop
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.label_list = label_list
        self.train_tfrecord_file = os.path.join(self.output_dir, "train.tf_record")

        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)

        if self.max_seq_length > self.bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self.max_seq_length, self.bert_config.max_position_embeddings))

        tf.gfile.MakeDirs(self.output_dir)

        self.processor = JustClassifierDataProcessor()
        self.processor.set_labels(self.label_list)

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)

        tpu_cluster_resolver = None

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        #distribution = tf.contrib.distribute.MirroredStrategy()
        self.run_config = tf.estimator.RunConfig(
            save_checkpoints_steps=self.save_checkpoints_steps,
            model_dir=self.output_dir
        )
        """
        self.run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=None,
            model_dir=self.output_dir,
            save_checkpoints_steps=self.save_checkpoints_steps,
            train_distribute=distribution,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.iterations_per_loop,
                num_shards=8,
                per_host_input_for_training=is_per_host))
        """

        self.num_train_steps = None
        self.num_warmup_steps = None
        if self.is_train:
            self.train_examples = self.processor.get_train_examples(self.train_file)
            self.num_train_steps = int(
                len(self.train_examples) / self.train_batch_size * self.num_train_epochs) or 1
            self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)

        model_fn = model_fn_builder(
            bert_config=self.bert_config,
            num_labels=len(label_list),
            init_checkpoint=self.init_checkpoint,
            learning_rate=self.learning_rate,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            use_tpu=False,
            use_one_hot_embeddings=False)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=self.run_config
        )
        """
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=self.run_config,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            predict_batch_size=self.predict_batch_size)
        """

    def export_savedmodel(self, save_dir):
        """

        :param save_dir: saved path for model
        :return:
        """
        tf.gfile.MakeDirs(save_dir)
        feature_spec = {
            "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64),
        }
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec=feature_spec)
        self.estimator._export_to_tpu = False
        self.estimator.export_saved_model(export_dir_base=save_dir, serving_input_receiver_fn=serving_input_fn)

    def export_savedmodel_rawinput(self, save_dir):
        feature_spec = {
            "input_text": tf.FixedLenFeature([], dtype=tf.string)
        }

        def serving_input_receiver_fn():
            """An input_fn that expects a serialized tf.Example."""
            from tensorflow.python.framework import dtypes
            from tensorflow.python.ops import array_ops
            from tensorflow.python.ops import parsing_ops
            from tensorflow.python.estimator.export.export import ServingInputReceiver

            serialized_tf_example = array_ops.placeholder(
                dtype=dtypes.string,
                shape=[None],
                name='input_example_tensor')
            receiver_tensors = {'examples': serialized_tf_example}
            features = parsing_ops.parse_example(serialized_tf_example, feature_spec)
            return ServingInputReceiver(features, receiver_tensors)

        return serving_input_receiver_fn

    def train_and_evaluate(self, eval_steps=1000, throttle_secs=600):

        if not tf.gfile.Exists(self.train_tfrecord_file):
            file_based_convert_examples_to_features(
                self.train_examples, self.label_list, self.max_seq_length, self.tokenizer, self.train_tfrecord_file)
        else:
            tf.logging.warning("{} 已经存在，不重新生成，如果想要重新生成，请先删除该文件".format(self.train_tfrecord_file))

        train_input_fn = file_based_input_fn_builder(
            input_file=self.train_tfrecord_file,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=True,
            batch_size=self.train_batch_size
        )
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=self.num_train_steps)
        eval_examples = self.processor.get_dev_examples(self.dev_file)
        eval_file = os.path.join(self.output_dir, "eval.tf_record")
        if not tf.gfile.Exists(eval_file):
            file_based_convert_examples_to_features(
                eval_examples, self.label_list, self.max_seq_length, self.tokenizer, eval_file)
        else:
            tf.logging.warning("{} 已经存在，不重新生成，如果想要重新生成，请先删除该文件".format(eval_file))
        eval_drop_remainder = False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            batch_size=self.eval_batch_size
        )
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps, throttle_secs=throttle_secs)
        tf.estimator.train_and_evaluate(estimator=self.estimator, train_spec=train_spec, eval_spec=eval_spec)

    def train(self):

        if not tf.gfile.Exists(self.train_tfrecord_file):
            file_based_convert_examples_to_features(
                self.train_examples, self.label_list, self.max_seq_length, self.tokenizer, self.train_tfrecord_file)
        else:
            tf.logging.warning("{} 已经存在，不重新生成，如果想要重新生成，请先删除该文件".format(self.train_tfrecord_file))

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(self.train_examples))
        tf.logging.info("  Batch size = %d", self.train_batch_size)
        tf.logging.info("  Num steps = %d", self.num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=self.train_tfrecord_file,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=True,
            batch_size=self.train_batch_size
        )

        self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)

    def eval(self):
        eval_examples = self.processor.get_dev_examples(self.dev_file)
        eval_file = os.path.join(self.output_dir, "eval.tf_record")
        if not tf.gfile.Exists(eval_file):
            file_based_convert_examples_to_features(
                eval_examples, self.label_list, self.max_seq_length, self.tokenizer, eval_file)
        else:
            tf.logging.warning("{} 已经存在，不重新生成，如果想要重新生成，请先删除该文件".format(eval_file))

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", self.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.

        eval_drop_remainder = False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            batch_size=self.eval_batch_size
        )

        result = self.estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    def predict(self, lines):
        if not isinstance(lines, list):
            lines = [lines]
        predict_examples = self.processor.get_test_examples_not_from_file(lines)

        features = convert_examples_to_features(predict_examples, self.label_list, self.max_seq_length, self.tokenizer)
        predict_input_fn = input_fn_builder(
            features=features,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False,
            batch_size=self.predict_batch_size
        )

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", self.predict_batch_size)

        result = self.estimator.predict(input_fn=predict_input_fn, yield_single_examples=True)
        tf.logging.info("{}".format(result))
        return result


def main():
    model = SimpleClassifierModel(
        is_train=True,
        bert_config_file="./data/chinese_L-12_H-768_A-12/bert_config.json",
        vocab_file="./data/chinese_L-12_H-768_A-12/vocab.txt",
        output_dir="./output",
        max_seq_length=512,
        train_file="./data/train.csv",
        dev_file="./data/dev.csv",
        init_checkpoint="./data/chinese_L-12_H-768_A-12/bert_model.ckpt",
        label_list=["0", "1", "2", "3"],
        num_train_epochs=5,
        train_batch_size=8
    )
    model.train()
    res = model.predict([["1", "李勇"], ["2", "保险"]])
    print("prediction is {}".format(list(res)))
    res = list(res)
    from sklearn.metrics.pairwise import cosine_similarity
    print(cosine_similarity([res[0]["output_layer"], res[1]["output_layer"]]))
    model.eval()
    model.export_savedmodel("./export")


if __name__ == "__main__":
    main()
