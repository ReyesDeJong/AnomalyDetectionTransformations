"""
Get MI images for a dataset and a respective transformation.
A MI image is defined as an image where every pixel represents the MI
 information of a NxN window in a dataset of images, wrt to the same region on a
 transformed dataset
"""

import os
import sys

import numpy as np

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.info_metrics.information_estimator_v2 import InformationEstimator
from parameters import loader_keys, general_keys
from modules.geometric_transform import transformations_tf
import tensorflow as tf
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
import time
from modules.utils import timer, plot_image, get_pvalue_welchs_ttest
import numpy as np
import matplotlib.pyplot as plt
from modules.geometric_transform.transformations_tf import makeGaussian, \
  cnn2d_depthwise_tf, check_shape_kernel
from scripts.mutual_info.new_ideas_tinkering. \
  creating_artificial_dataset import CirclesFactory


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


class MIImageCalculator(object):
  def __init__(self, information_estimator: InformationEstimatorByBatch,
      window_size, window_stride=None):
    self.information_estimator = information_estimator
    self.window_size = window_size
    self.window_stride = window_stride
    # TODO: Not used
    self.padding = 0
    if window_stride == None:
      self.window_stride = window_size

  # TODO: method to normilize images

  # TODO: include_padding
  def image_array_to_patches(self, image_array):
    n_images = len(image_array)
    images_size = image_array.shape[1]
    n_windows_in_an_image = self.get_how_many_windows_fit_in_a_single_image(
        images_size, self.window_size, self.window_stride, self.padding)
    # print(n_windows_in_an_image)
    n_windows_in_a_row = int(np.sqrt(n_windows_in_an_image))
    list_of_patches_for_every_image = []
    for img_idx in range(n_images):
      image = image_array[img_idx]
      list_of_patches_for_a_single_image = []
      for row_idx in range(n_windows_in_a_row):
        for col_idx in range(n_windows_in_a_row):
          # print(row_idx * (1 + (self.window_stride-1)), row_idx * (
          #     1 + (self.window_stride-1)) + self.window_size)
          window = image[row_idx * (1 + (self.window_stride - 1)):row_idx * (
              1 + (self.window_stride - 1)) + self.window_size,
                   col_idx * (1 + (self.window_stride - 1)):col_idx * (
                       1 + (self.window_stride - 1)) + self.window_size, ...]
          # print(window.shape)

          list_of_patches_for_a_single_image.append(window)

      list_of_patches_for_every_image.append(list_of_patches_for_a_single_image)

    return np.array(list_of_patches_for_every_image)

  def get_how_many_windows_fit_in_a_single_image(self, image_size, window_size,
      stride, padding):
    n_windows_in_a_row = ((image_size - window_size + 2 * padding) / stride) + 1
    n_windows_in_a_row = np.ceil(n_windows_in_a_row)
    return int(np.square(n_windows_in_a_row))

  def calculate_mi_for_patches(self, patches_array_X, patches_array_Y):
    n_patches_per_sample = patches_array_Y.shape[1]
    mi_for_every_patch = []
    for patch_idx in range(n_patches_per_sample):
      patch_x = patches_array_X[:, patch_idx, ...]
      patch_y = patches_array_Y[:, patch_idx, ...]
      mutual_info_estimation = self.information_estimator. \
        mutual_information_by_batch(patch_x, patch_y)
      mi_for_every_patch.append(mutual_info_estimation)
    return np.array(mi_for_every_patch)

  def mi_images(self, X, Y):
    patches_X = self.image_array_to_patches(X)
    patches_Y = self.image_array_to_patches(Y)
    mi_of_patches = self.calculate_mi_for_patches(patches_X, patches_Y)
    image_size = int(np.sqrt(len(mi_of_patches)))
    # print(mi_of_patches[:image_size])
    mi_images = mi_of_patches.reshape((image_size, image_size, -1))
    # print(mi_images[0, :])
    return mi_images


if __name__ == '__main__':
  SHOW_PLOTS = True
  N_IMAGES = 7000
  WINDOW_SIZE = 3
  SIGMA_ZERO = 2.0
  BATCH_SIZE = 512
  TRANSFORMATION_SHIFT = 8
  circle_factory = CirclesFactory()
  mi_estimator = InformationEstimatorByBatch(SIGMA_ZERO, BATCH_SIZE)
  mi_image_calculator = MIImageCalculator(information_estimator=mi_estimator,
                                          window_size=WINDOW_SIZE)
  images = circle_factory.get_final_dataset(N_IMAGES)
  images_transformed = np.roll(images, shift=TRANSFORMATION_SHIFT, axis=2)
  circle_factory.plot_n_images(images, plot_show=SHOW_PLOTS, title='X')
  circle_factory.plot_n_images(images_transformed, plot_show=SHOW_PLOTS,
                               title='T(X)')
  mi_images = mi_image_calculator.mi_images(images, images_transformed)
  print(mi_images.shape)
  circle_factory.plot_n_images(mi_images, plot_show=SHOW_PLOTS,
                               title='MI(X, T(X))')
  mean_mi_image = np.mean(mi_images, axis=-1)
  plt.imshow(mean_mi_image)
  if SHOW_PLOTS:
    plt.show()
  plt.close()
  std_mi_image = np.std(mi_images, axis=-1)
  plt.imshow(std_mi_image)
  if SHOW_PLOTS:
    plt.show()
  plt.close()
  print('')
