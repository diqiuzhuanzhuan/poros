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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from poros.bert_model import modeling
from poros.bert_model import optimization
import tensorflow as tf

_flag = dict()

## Required parameters

_flag["bert_config"] = None
_flag["input_file"] = None
_flag["output_dir"] = None

_flag["log_step_count_steps"] = 100

"""
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
"""

"""
flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")
"""

"""
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")
"""
_flag["output_dir"] = None

## Other parameters
"""
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")
"""
_flag["init_checkpoint"] = None

"""
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")
"""
_flag["max_seq_length"] = 128

"""
flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")
"""
_flag["max_predictions_per_seq"] = 20

#flags.DEFINE_bool("do_train", False, "Whether to run training.")
_flag["do_train"] = False

#flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
_flag["do_eval"] = False

#flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
_flag["train_batch_size"] = 32

#flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
_flag["eval_batch_size"] = 8

#flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
_flag["learning_rate"] = 5e-5

#flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")
_flag["num_train_steps"] = 100000

#flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")
_flag["num_warmup_steps"] = 10000

"""
flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")
"""
_flag["save_checkpoints_steps"] = 1000

"""
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")
"""
_flag["iterations_per_loop"] = 1000

#flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")
_flag["max_eval_steps"] = 100

#flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
_flag["use_tpu"] = False

"""tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
"""
_flag["tpu_name"] = None

"""
tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
"""
_flag["tpu_zone"] = None

#tf.flags.DEFINE_string(
#    "gcp_project", None,
#    "[Optional] Project name for the Cloud TPU-enabled project. If not "
#    "specified, we will attempt to automatically detect the GCE project from "
#    "metadata.")
_flag["gcp_project"] = None

#tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
_flag["master"] = None

#flags.DEFINE_integer(
#    "num_tpu_cores", 8,
#    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

_flag["num_tpu_cores"] = 8

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
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
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            bert_config, model.get_sequence_output(), model.get_embedding_table(),
            masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
            bert_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = _flag["alpha"] * masked_lm_loss + _flag["beta"] * next_sentence_loss

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
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                          masked_lm_weights, next_sentence_example_loss,
                          next_sentence_log_probs, next_sentence_labels):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                                 [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                next_sentence_log_probs = tf.reshape(
                    next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
                next_sentence_predictions = tf.argmax(
                    next_sentence_log_probs, axis=-1, output_type=tf.int32)
                next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
                next_sentence_accuracy = tf.metrics.accuracy(
                    labels=next_sentence_labels, predictions=next_sentence_predictions)
                next_sentence_mean_loss = tf.metrics.mean(
                    values=next_sentence_example_loss)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                    "next_sentence_accuracy": next_sentence_accuracy,
                    "next_sentence_loss": next_sentence_mean_loss,
                }

            eval_metrics = (metric_fn, [
                masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                masked_lm_weights, next_sentence_example_loss,
                next_sentence_log_probs, next_sentence_labels
            ])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "input_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.FixedLenFeature([1], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


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


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not _flag["do_train"] and not _flag["do_eval"]:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(_flag["bert_config_file"])

    tf.gfile.MakeDirs(_flag["output_dir"])

    input_files = []
    for input_pattern in _flag["input_file"].split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    tpu_cluster_resolver = None
    if _flag["use_tpu"] and _flag["tpu_name"]:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            _flag["tpu_name"], zone=_flag["tpu_zone"], project=_flag["gcp_project"])

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=_flag["master"],
        model_dir=_flag["output_dir"],
        save_checkpoints_steps=_flag["save_checkpoints_steps"],
        log_step_count_steps=_flag["log_step_count_steps"],
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=_flag["iterations_per_loop"],
            num_shards=_flag["num_tpu_cores"],
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=_flag["init_checkpoint"],
        learning_rate=_flag["learning_rate"],
        num_train_steps=_flag["num_train_steps"],
        num_warmup_steps=_flag["num_warmup_steps"],
        use_tpu=_flag["use_tpu"],
        use_one_hot_embeddings=_flag["use_tpu"])

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=_flag["use_tpu"],
        model_fn=model_fn,
        config=run_config,
        train_batch_size=_flag["train_batch_size"],
        eval_batch_size=_flag["eval_batch_size"])

    if _flag["do_train"]:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", _flag["train_batch_size"])
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=_flag["max_seq_length"],
            max_predictions_per_seq=_flag["max_predictions_per_seq"],
            is_training=True)
        estimator.train(input_fn=train_input_fn, max_steps=_flag["num_train_steps"])

    if _flag["do_eval"]:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", _flag["eval_batch_size"])

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=_flag["max_seq_length"],
            max_predictions_per_seq=_flag["max_predictions_per_seq"],
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=_flag["max_eval_steps"])

        output_eval_file = os.path.join(_flag["output_dir"], "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


def run(input_file, bert_config_file, output_dir, max_seq_length=128, max_predictions_per_seq=20, do_train=False, do_eval=True,
        train_batch_size=32, eval_batch_size=8, learning_rate=5e-5, num_train_steps=100000, num_warmup_steps=1000,
        init_checkpoint=None, save_checkpoints_steps=1000, iterations_per_loop=1000, max_eval_steps=100, log_step_count_steps=100, alpha=1.0, beta=1.0):
    _flag["input_file"] = input_file
    _flag["bert_config_file"] = bert_config_file
    _flag["output_dir"] = output_dir
    _flag["max_seq_length"] = max_seq_length
    _flag["max_predictions_per_seq"] = max_predictions_per_seq
    _flag["do_train"] = do_train
    _flag["do_eval"] = do_eval
    _flag["train_batch_size"] = train_batch_size
    _flag["eval_batch_size"] = eval_batch_size
    _flag["learning_rate"] = learning_rate
    _flag["num_train_steps"] = num_train_steps
    _flag["num_warmup_steps"] = num_warmup_steps
    _flag["save_checkpoints_steps"] = save_checkpoints_steps
    _flag["iterations_per_loop"] = iterations_per_loop
    _flag["max_eval_steps"] = max_eval_steps
    _flag["log_step_count_steps"] = log_step_count_steps
    _flag["alpha"] = alpha
    _flag["beta"] = beta
    _flag["init_checkpoint"] = init_checkpoint
    main(None)


if __name__ == "__main__":
    run(input_file="./data/output", bert_config_file="./test_data/bert_config.json", num_train_steps=100, output_dir="./output", do_train=True)
