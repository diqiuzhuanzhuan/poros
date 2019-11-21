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


def init_from_checkpoint(init_checkpoint, tvars):
    """
    use variables in init_checkpoint to set value of variables in tvars

    Args:
        init_checkpoint: a checkpoint file
        tvars: a list of Variables, a Tensor
    """
    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        checkpoint_vars_name = assignment_map.keys()
        checkpoint_vars = load_variables(init_checkpoint, checkpoint_vars_name)
        count = 0
        for tvar in tvars:
            if tvar.name.endswith(":0"):
                tvar_name = tvar.name[:-2]
            else:
                tvar_name = tvar.name
            if tvar_name not in checkpoint_vars:
                continue
            tf.keras.backend.set_value(tvar, checkpoint_vars[tvar_name])
            count += 1
            init_string = ", *INIT_FROM_CKPT*"
            tf.get_logger().info("  name = %s, shape = %s%s", tvar.name, tvar.shape, init_string)
        tf.get_logger().info("init {} variables.".format(count))


def restore(init_checkpoint):
    """
    restore variables from a checkpoint file.
    note that it doesn't work while in tf.keras model
    """
    tvars = tf.compat.v1.trainable_variables()
    assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
    tf.get_logger().info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.get_logger().info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
