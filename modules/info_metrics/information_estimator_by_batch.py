"""
MI estimation of a whole dataset on a batch basis
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.info_metrics.information_estimator_v2 import InformationEstimator
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class InformationEstimatorByBatch(InformationEstimator):
  def __init__(self, sigma_zero, batch_size, random_seed=42,
      shuffle_buffer_size=10000, x_and_y_as_images=True):
    super().__init__(sigma_zero=sigma_zero)
    self.batch_size = batch_size
    self.random_seed = random_seed
    self.shuffle_buffer_size = shuffle_buffer_size
    self.x_is_image = x_and_y_as_images
    self.y_is_image = x_and_y_as_images

  def mutual_information_by_batch(self, X, Y):
    estimation_ds = tf.data.Dataset.from_tensor_slices(
        (X, Y)).shuffle(self.shuffle_buffer_size, seed=self.random_seed).batch(
        self.batch_size)
    mi_list = []
    for x_batch, y_batch in estimation_ds:
      # print(tf.shape(x_batch))
      # print(tf.shape(y_batch))
      # diff = x_batch-y_batch
      # print(np.unique(diff.numpy()))
      mi_estimation = self.mutual_information(
          x_batch, y_batch, x_is_image=self.x_is_image,
          y_is_image=self.y_is_image)
      mi_list.append(mi_estimation.numpy())
    return np.array(mi_list)

  def mutual_information_mean_fast(self, X, Y, x_is_image=True,
      y_is_image=True):
    estimation_ds = tf.data.Dataset.from_tensor_slices(
        (X, Y)).shuffle(self.shuffle_buffer_size, seed=self.random_seed).batch(
        self.batch_size)
    mean_mi = tf.keras.metrics.Mean(name='MI')
    for x_batch, y_batch in estimation_ds:
      mi_estimation = self.mutual_information(
          x_batch, y_batch, x_is_image=self.x_is_image,
          y_is_image=self.y_is_image)
      mean_mi(mi_estimation)
    return mean_mi.result()
