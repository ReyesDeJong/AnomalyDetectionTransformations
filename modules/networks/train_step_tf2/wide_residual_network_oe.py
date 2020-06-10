from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from modules.networks.train_step_tf2.wide_residual_network import \
  WideResidualNetwork

WEIGHT_DECAY = 0.5 * 0.0005


# TODO: data_format to constructors of layers
class WideResidualNetworkOE(WideResidualNetwork):
  def __init__(self, input_shape, n_classes, depth,
      widen_factor=1, dropout_rate=0.0, final_activation='softmax',
      name='wide_resnet_oe', data_format='channels_last',
      weight_decay=WEIGHT_DECAY, model_path='', lambda_oe=1.0):
    super().__init__(input_shape, n_classes, depth,
                     widen_factor, dropout_rate, final_activation,
                     name, data_format,
                     weight_decay, model_path)
    self.lambda_oe = lambda_oe

  def _init_builds(self):
    # builds
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.Adam()
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    self.eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    self.eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='eval_accuracy')

  @tf.function
  def train_step(self, images, labels):
    with tf.GradientTape() as tape:
      predictions = self.call(images, training=True)
      inlier_mask = labels != - 99
      outlier_mask = ~ inlier_mask
      # print(predictions[outlier_mask])
      # print(labels[outlier_mask])
      loss = self.loss_object(labels[inlier_mask], predictions[inlier_mask]) - \
             self.lambda_oe * tf.reduce_mean(
        tf.math.log(predictions[outlier_mask]))
    gradients = tape.gradient(loss, self.trainable_variables)
    self.train_loss(loss)
    self.train_accuracy(labels, predictions)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

  # # @tf.function
  # def train_step(self, images, labels):
  #   inlier_mask = labels != - 99
  #   print(inlier_mask)
  #   with tf.GradientTape() as tape:
  #     predictions = self.call(images, training=True)
  #     loss = self.loss_object(labels, predictions)
  #   gradients = tape.gradient(loss, self.trainable_variables)
  #   self.train_loss(loss)
  #   self.train_accuracy(labels, predictions)
  #   self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
