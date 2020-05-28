"""
Get MI images for a dataset and a respective transformation.
A MI image is defined as an image where every pixel represents the MI
 information of a NxN window in a dataset of images, wrt to the same region on a
 transformed dataset
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.info_metrics.information_estimator_by_batch import \
  InformationEstimatorByBatch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.transform_selection.mutual_info.mi_images_on_transformations_manager_selector import MIIOnTransformationsManager
from tqdm import tqdm


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

  def mi_images(self, X, Y, normalize_patches=False):
    patches_X = self.image_array_to_patches(X)
    # print(patches_X)
    patches_Y = self.image_array_to_patches(Y)
    if normalize_patches:
      patches_X = self.normalize_patches_1_1(patches_X)
      patches_Y = self.normalize_patches_1_1(patches_Y)

    # print(patches_X)
    mi_of_patches = self.calculate_mi_for_patches(patches_X, patches_Y)
    image_size = int(np.sqrt(len(mi_of_patches)))
    # print(mi_of_patches[:image_size])
    mi_images = mi_of_patches.reshape((image_size, image_size, -1))
    mi_images = np.rollaxis(mi_images, -1)
    # print(mi_images[0, :])
    return mi_images

  def mi_images_mean(self, X, Y, normalize_patches=False):
    mi_images = self.mi_images(X, Y, normalize_patches)
    return np.mean(mi_images, axis=0)

  def mi_images_std(self, X, Y, normalize_patches=False):
    mi_images = self.mi_images(X, Y, normalize_patches)
    return np.std(mi_images, axis=0)

  def normalize_patches_1_1(self, patches):
    patches -= np.nanmin(patches, axis=(2, 3))[:, :, np.newaxis, np.newaxis, :]
    patches = patches / self._replace_zeros_with_ones(
        np.nanmax(patches, axis=(2, 3))[:, :, np.newaxis, np.newaxis, :])
    patches = 2 * patches - 1
    return patches

  def _replace_zeros_with_ones(self, array):
    array[array == 0] = 1
    return array

  def mii_for_transformations(self, X, transformer: AbstractTransformer,
      normalize_patches=False, plot_transformations=False) -> MIIOnTransformationsManager:
    n_transformations = transformer.n_transforms
    mii_images = {}
    transformation_tuples = transformer.transformation_tuples
    # print(transformation_tuples)
    print('Calculating MII for all %i transformations' % int(n_transformations))
    for trnsform_idx in tqdm(range(n_transformations)):
      current_transformation_tuple = transformation_tuples[trnsform_idx]
      x_transformed, _ = transformer.apply_transforms(X, [trnsform_idx])
      if plot_transformations:
        CirclesFactory().plot_n_images(
            x_transformed, plot_show=True,
            title=str(trnsform_idx) + '_' + str(current_transformation_tuple))

      mii_images_specific_transform = self.mi_images(X, x_transformed,
                                                     normalize_patches)
      mii_images[current_transformation_tuple] = mii_images_specific_transform

    return MIIOnTransformationsManager(mii_images, transformer)


if __name__ == '__main__':
  from modules.data_loaders.artificial_dataset_factory import CirclesFactory

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  SHOW_PLOTS = False
  N_IMAGES = 64 * 4
  WINDOW_SIZE = 3
  SIGMA_ZERO = 2.0
  BATCH_SIZE = 64
  TRANSFORMATION_SHIFT = 6
  circle_factory = CirclesFactory()
  mi_estimator = InformationEstimatorByBatch(SIGMA_ZERO, BATCH_SIZE)
  mi_image_calculator = MIImageCalculator(information_estimator=mi_estimator,
                                          window_size=WINDOW_SIZE)
  images = circle_factory.get_final_dataset(N_IMAGES)
  images_transformed = np.roll(images, shift=TRANSFORMATION_SHIFT, axis=2)
  # images_transformed = np.rot90(images, axes=(1, 2))
  circle_factory.plot_n_images(images, plot_show=SHOW_PLOTS, title='X')
  circle_factory.plot_n_images(images_transformed, plot_show=SHOW_PLOTS,
                               title='T(X)')
  # images = np.zeros_like(images)
  # print(images.shape)
  # print(np.unique(np.min(images, axis=(-1,-2,-3))))
  # images = np.random.normal(0,1,images.shape)
  mi_images = mi_image_calculator.mi_images(images, images_transformed)
  print(mi_images.shape)
  circle_factory.plot_n_images(mi_images, plot_show=SHOW_PLOTS,
                               title='MI(X, T(X))')
  mean_mi_image = np.mean(mi_images, axis=0)
  plt.imshow(mean_mi_image)
  plt.colorbar()
  if SHOW_PLOTS:
    plt.show()
  plt.close()
  std_mi_image = np.std(mi_images, axis=0)
  plt.imshow(std_mi_image)
  plt.colorbar()
  if SHOW_PLOTS:
    plt.show()
  plt.close()
  print('')
