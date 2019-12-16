"""
HiTS stamps outlier loader

safe max data loading float64 (~50GB): (by using float 32 it is reduced to half)
(3500000, 21, 21, 4) == (140*25000, 21, 21, 4)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.data_set_generic import Dataset
from parameters import general_keys, param_keys
from modules.geometric_transform.transformations_tf import AbstractTransformer
from parameters import loader_keys
from modules.data_loaders.hits_loader import HiTSLoader
from modules import utils


# TODO: refactor to integrate with ZTF dataset and
#  an easy coupling with other classic datasets
# Todo: Do some refactoring to include kwargs
class HiTSOutlierLoader(object):

  def __init__(self, params: dict, dataset_name='hits'):
    # n_samples_by_class = 10000, test_size = 0.20, val_size = 0.10,
    # return_val = False, channels_to_get = [0, 1, 2, 3]
    self.n_samples_by_class = params[loader_keys.N_SAMPLES_BY_CLASS]
    self.test_percentage_all_data = params[loader_keys.TEST_PERCENTAGE]
    self.data_path = params[loader_keys.DATA_PATH]
    self.val_inlier_percentage = params[loader_keys.VAL_SET_INLIER_PERCENTAGE]
    self.used_channels = params[loader_keys.USED_CHANNELS]
    self.random_seed = params[general_keys.RANDOM_SEED]
    self.crop_size = params[loader_keys.CROP_SIZE]
    self.template_save_path = self._get_template_save_path()
    self.name = dataset_name + '_%i_channels' % len(self.used_channels)

  # TODO: code things to iteratively add text at begining of getters
  def _get_template_save_path(self) -> str:
    """get name of final saved file to check if it's been already generated"""
    text_to_add = 'seed%i_crop%s_nChannels%i' % (
      self.random_seed, str(self.crop_size), len(self.used_channels))
    return utils.add_text_to_beginning_of_file_path(self.data_path, text_to_add)

  def get_unsplitted_dataset(self) -> Dataset:
    """get preprocessed dataset, prior to outlier-inlier splitting"""
    # check if preprocessing has already been done
    unsplitted_data_path = utils.add_text_to_beginning_of_file_path(
        self.template_save_path, 'unsplitted')
    if os.path.exists(unsplitted_data_path):
      return pd.read_pickle(unsplitted_data_path)
    # params for hits loader, it performs an by sample 0-1 norm that I think
    # is not useful, becuse data is already 0-1
    params = {
      param_keys.DATA_PATH_TRAIN: self.data_path,
      param_keys.BATCH_SIZE: 0
    }
    # TODO: check None thing, labels value is None
    #  because it is not used, retrieving both labels
    data_loader = HiTSLoader(params, label_value=None,
                             first_n_samples_by_class=self.n_samples_by_class,
                             test_size=None, validation_size=None,
                             channels_to_get=self.used_channels)
    dataset = data_loader.get_single_dataset()
    utils.save_pickle(dataset, unsplitted_data_path)
    return dataset

  def get_preprocessed_unsplitted_dataset(self):
    preproc_data_path = utils.add_text_to_beginning_of_file_path(
        self.template_save_path, 'preproc')
    if os.path.exists(preproc_data_path):
      return pd.read_pickle(preproc_data_path)

    dataset = self.get_unsplitted_dataset()
    # preprocessing -1 to 1 normalize
    dataset.data_array = 2 * (
        dataset.data_array / np.max(dataset.data_array)) - 1
    utils.save_pickle(dataset, preproc_data_path)
    return dataset

  # TODO: check what happens inside here, particularly correct label consistency when new_labels are defined
  def get_outlier_detection_datasets(self):
    """get outlier trainval test sets, by slecting class 0 as outliers (bogus in Hitd) and generating a train-val
    set of only inliers, while a test set with half-half inliers and outliers"""
    outlier_data_path = utils.add_text_to_beginning_of_file_path(
        self.template_save_path, 'outlier')
    if os.path.exists(outlier_data_path):
      return pd.read_pickle(outlier_data_path)

    dataset = self.get_preprocessed_unsplitted_dataset()
    # labels from 5 classes to 0-1 as bogus-real
    bogus_class_indx = 0
    new_labels = (dataset.data_label.flatten() != bogus_class_indx) * 1.0
    print(np.mean(new_labels == dataset.data_label))
    # print(np.unique(new_labels, return_counts=True))
    inlier_task = 1  #
    n_outliers = int(
      np.round(self.test_percentage_all_data * self.n_samples_by_class))
    # separate data into train-val-test
    outlier_indexes = np.where(new_labels != inlier_task)[0]
    np.random.RandomState(seed=self.random_seed).shuffle(outlier_indexes)
    test_outlier_idxs = outlier_indexes[:n_outliers]
    inlier_indexes = np.where(new_labels == inlier_task)[0]
    # real == inliers
    val_size_inliers = int(np.round(np.sum(
        (new_labels == inlier_task)) * self.val_inlier_percentage))
    print(val_size_inliers)
    np.random.RandomState(seed=self.random_seed).shuffle(inlier_indexes)
    train_inlier_idxs = inlier_indexes[val_size_inliers:]
    val_inlier_idxs = inlier_indexes[:val_size_inliers]
    # train-test inlier indexes
    train_inlier_idxs = train_inlier_idxs[n_outliers:]
    test_inlier_idxs = train_inlier_idxs[:n_outliers]

    X_train, y_train = dataset.data_array[train_inlier_idxs], new_labels[
      train_inlier_idxs]
    X_val, y_val = dataset.data_array[val_inlier_idxs], new_labels[
      val_inlier_idxs]
    X_test, y_test = np.concatenate(
        [dataset.data_array[test_inlier_idxs],
         dataset.data_array[test_outlier_idxs]]), np.concatenate(
        [new_labels[test_inlier_idxs], new_labels[test_outlier_idxs]])
    print('train: ', np.unique(y_train, return_counts=True))
    print('val: ', np.unique(y_val, return_counts=True))
    print('test: ', np.unique(y_test, return_counts=True))
    sets_tuple = ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    utils.save_pickle(sets_tuple, outlier_data_path)
    return sets_tuple

  # TODO: implement transformation loading
  def get_transformed_datasets(self, transformer: AbstractTransformer):
    """transform daa and save to avoid doin it over and over again"""
    transformed_data_path = utils.add_text_to_beginning_of_file_path(
        self.template_save_path, '%s_outlier' % transformer.name)
    if os.path.exists(transformed_data_path):
      return pd.read_pickle(transformed_data_path)
    (x_train, y_train), (x_val, y_val), (
      x_test, y_test) = self.get_outlier_detection_datasets()
    print('train: ', np.unique(y_train, return_counts=True))
    print('val: ', np.unique(y_val, return_counts=True))
    print('test: ', np.unique(y_test, return_counts=True))
    x_train_transformed, train_transform_inds = \
      transformer.apply_all_transforms(x_train)
    x_val_transformed, val_transform_inds = \
      transformer.apply_all_transforms(x_val)
    x_test_transformed, test_transform_inds = \
      transformer.apply_all_transforms(x_test)
    sets_tuple = ((x_train_transformed, train_transform_inds),
                  (x_val_transformed, val_transform_inds),
                  (x_test_transformed, test_transform_inds))
    utils.save_pickle(sets_tuple, transformed_data_path)
    return sets_tuple


if __name__ == "__main__":
  from modules.geometric_transform.transformations_tf import Transformer
  import datetime
  import time

  start_time = time.time()
  params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2, 3],
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  transformer = Transformer()
  hits_outlier_dataset = HiTSOutlierLoader(params)

  dataset = hits_outlier_dataset.get_unsplitted_dataset()
  print('dataset: ', np.unique(dataset.data_label, return_counts=True))
  (X_train, y_train), (X_val, y_val), (
    X_test, y_test) = hits_outlier_dataset.get_outlier_detection_datasets()
  print('train: ', np.unique(y_train, return_counts=True))
  print('val: ', np.unique(y_val, return_counts=True))
  print('test: ', np.unique(y_test, return_counts=True))
  (X_train_trans, y_train_trans), (X_val_trans, y_val_trans), (
    X_test_trans, y_test_trans) = hits_outlier_dataset.get_transformed_datasets(
      transformer)
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time usage %s: %s" % (transformer.name, str(time_usage)), flush=True)
  print('train: ', np.unique(y_train_trans, return_counts=True))
  print('val: ', np.unique(y_val_trans, return_counts=True))
  print('test: ', np.unique(y_test_trans, return_counts=True))
