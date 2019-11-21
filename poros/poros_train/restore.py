# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import tensorflow as tf
import collections
import re


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def load_variables(ckpt_dir_or_file, names):
    """Returns the tensor value of the given variable in the checkpoint.

    Args:
      ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.
      name: Names of the variable to return.

    Returns:
      A dict containing all variables corresponding to names, each in it with a copy of the value of this variable.
      ```
        {'bert/encoder/bias': var1}
      ```
    """
    # TODO(b/29227106): Fix this in the right place and remove this.
    name_to_vars = collections.OrderedDict()
    reader = tf.train.load_checkpoint(ckpt_dir_or_file)
    for name in names:
        if name.endswith(":0"):
            name = name[:-2]
        name_to_vars[name] = reader.get_tensor(name)

    return name_to_vars


def restore(init_checkpoint):
    tvars = tf.trainable_variables()
    assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
