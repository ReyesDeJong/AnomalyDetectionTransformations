from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from parameters import general_keys
import numpy as np


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

def accuracies_by_threshold(labels, scores, thresholds):
  inliers_scores = scores[labels == 1]
  outliers_scores = scores[labels != 1]

  accuracies = []
  for thr in thresholds:
    FP = np.sum(outliers_scores > thr)
    TP = np.sum(inliers_scores > thr)
    TN = np.sum(outliers_scores <= thr)
    FN = np.sum(inliers_scores <= thr)

    accuracy = (TP + TN) / (FP + TP + TN + FN)
    accuracies.append(accuracy)
  return accuracies

def accuracies_by_thresholdv2(labels, scores, thresholds):
  accuracies = []
  for thr in thresholds:
    accuracy = accuracy_at_thr(labels, scores, thr)
    accuracies.append(accuracy)
  return accuracies

def accuracy_at_thr(labels, scores, threshold):
  inliers_scores = scores[labels == 1]
  outliers_scores = scores[labels != 1]
  scores_preds = (np.concatenate(
      [inliers_scores, outliers_scores]) > threshold) * 1
  accuracy_at_thr = np.mean(scores_preds == np.concatenate(
      [np.ones_like(inliers_scores), np.zeros_like(outliers_scores)]))
  return accuracy_at_thr
