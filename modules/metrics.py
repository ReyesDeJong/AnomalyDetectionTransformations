from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from parameters import general_keys


# TODO: transform metrics to a class in order to easily return metric dict

def accuracy(labels, predictions, is_onehot=False):
  with tf.name_scope('accuracy'):
    if is_onehot:
      labels = tf.argmax(labels, axis=-1)
      predictions = tf.argmax(predictions, axis=-1)
    correct_predictions = tf.equal(labels, predictions)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
  metrics_dict = {general_keys.ACCURACY: accuracy}
  return metrics_dict


def flatten(inputs, name=None):
  """ Flattens [batch_size, d0, ..., dn] to [batch_size, d0*...*dn]
  """
  with tf.name_scope(name):
    dim = tf.reduce_prod(tf.shape(inputs)[1:])
    outputs = tf.reshape(inputs, shape=(-1, dim))
  return outputs