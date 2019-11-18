"""
ZTF stamps outlier loader
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.data_set_generic import Dataset
from parameters import general_keys, param_keys
from transformations import AbstractTransformer
from parameters import loader_keys
from modules.data_loaders.frame_to_input import FrameToInput
from modules import utils


class ZTFOutlierLoader(object):

  def __init__(self, params: dict, transformer: AbstractTransformer = None):
    self.transformer = transformer
    self.data_path = params[loader_keys.DATA_PATH]
    self.val_inlier_percentage = params[loader_keys.VAL_SET_INLIER_PERCENTAGE]
    self.used_channels = params[loader_keys.USED_CHANNELS]
    self.crop_size = params[loader_keys.CROP_SIZE]
    self.random_seed = params[general_keys.RANDOM_SEED]
    self.transform_inlier_class_value = params[
      loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE]
    self.preprocessed_data_path = self._get_preprocessed_data_path()

  def _get_preprocessed_data_path(self) -> str:
    """get name of final saved file to check if it's been already generated"""
    text_to_add = 'preprocessed_seed%i_crop%i_nChannels%i' % (
      self.random_seed, self.crop_size, len(self.used_channels))
    return utils.add_text_to_beginning_of_file_path(self.data_path, text_to_add)

  def get_unsplitted_dataset(self) -> Dataset:
    """get preprocessed dataset, prior to outlier-inlier splitting"""
    # check if preprocessing has already been done
    unsplitted_data_path = utils.add_text_to_beginning_of_file_path(
        self.preprocessed_data_path, 'unsplitted')
    if os.path.exists(unsplitted_data_path):
      return pd.read_pickle(unsplitted_data_path)
    # useful to avoid frameToInput to perform the transformation of Dataframe to a pickle dict
    data_path = self.data_path
    unprocessed_unsplitted_data_path = utils.add_text_to_beginning_of_file_path(
        self.data_path, 'unprocessed_unsplitted')
    if os.path.exists(unprocessed_unsplitted_data_path):
      data_path = unprocessed_unsplitted_data_path

    # params for Frame input ztf loader and preprocessing
    params = {
      param_keys.DATA_PATH_TRAIN: data_path,
      param_keys.BATCH_SIZE: 0,
      param_keys.CHANNELS_TO_USE: self.used_channels,
      param_keys.TEST_SIZE: 0,  # not used
      param_keys.VAL_SIZE: 0,  # not used
      param_keys.NANS_TO: 0,
      param_keys.CROP_SIZE: self.crop_size,
      param_keys.CONVERTED_DATA_SAVEPATH: unprocessed_unsplitted_data_path
    }
    # instantiate loader, set preprocessor, load dataset
    data_loader = FrameToInput(params)
    data_loader.dataset_preprocessor.set_pipeline(
        [data_loader.dataset_preprocessor.check_single_image,
         data_loader.dataset_preprocessor.clean_misshaped,
         data_loader.dataset_preprocessor.select_channels,
         data_loader.dataset_preprocessor.normalize_by_image,
         data_loader.dataset_preprocessor.nan_to_num,
         data_loader.dataset_preprocessor.crop_at_center
         ])
    dataset = data_loader.get_single_dataset()
    utils.save_pickle(dataset, unsplitted_data_path)
    return dataset

  # TODO: check what happens inside here, particularly correct label consistency when new_labels are defined
  def get_outlier_detection_datasets(self):
    """get outlier trainval test sets, by slecting class 4 as outliers (bogus in ZTF) and generating a train-val
    set of only inliers, while a test set with half-half inliers and outliers"""
    outlier_data_path = utils.add_text_to_beginning_of_file_path(
        self.preprocessed_data_path, 'outlier')
    if os.path.exists(outlier_data_path):
      return pd.read_pickle(outlier_data_path)

    dataset = self.get_unsplitted_dataset()
    # labels from 5 classes to 0-1 as bogus-real
    bogus_class_indx = 4
    new_labels = (dataset.data_label.flatten() != bogus_class_indx) * 1.0
    # print(np.unique(new_labels, return_counts=True))
    inlier_task = 1  #
    # separate data into train-val-test
    outlier_indexes = np.where(new_labels != inlier_task)[0]
    inlier_indexes = np.where(new_labels == inlier_task)[0]
    # real == inliers
    val_size_inliers = int(np.round(np.sum(
        (new_labels == inlier_task)) * self.val_inlier_percentage))
    np.random.RandomState(seed=self.random_seed).shuffle(inlier_indexes)
    # train-val indexes inlier indexes
    train_inlier_idxs = inlier_indexes[val_size_inliers:]
    val_inlier_idxs = inlier_indexes[:val_size_inliers]
    # train-test inlier indexes
    n_outliers = np.sum(new_labels != inlier_task)
    train_inlier_idxs = train_inlier_idxs[n_outliers:]
    test_inlier_idxs = train_inlier_idxs[:n_outliers]

    X_train, y_train = dataset.data_array[train_inlier_idxs], new_labels[
      train_inlier_idxs]
    X_val, y_val = dataset.data_array[val_inlier_idxs], new_labels[
      val_inlier_idxs]
    X_test, y_test = np.concatenate(
        [dataset.data_array[test_inlier_idxs],
         dataset.data_array[outlier_indexes]]), np.concatenate(
        [new_labels[test_inlier_idxs], new_labels[outlier_indexes]])
    # print('train: ', np.unique(y_train, return_counts=True))
    # print('val: ', np.unique(y_val, return_counts=True))
    # print('test: ', np.unique(y_test, return_counts=True))
    sets_tuple = ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    utils.save_pickle(sets_tuple, outlier_data_path)
    return sets_tuple

  def _perform_transform(self, x):
    """generate transform inds, that are the labels of each transform and
    its respective transformed data"""
    transform_inds = np.tile(
        np.arange(self.transformer.n_transforms), len(x))
    x_transformed = self.transformer.transform_batch(
        np.repeat(x, self.transformer.n_transforms, axis=0),
        transform_inds)
    return x_transformed, transform_inds

  # TODO: implement transformation loading
  def get_transformed_datasets(self):
    """transform daa and save to avoid doin it over and over again"""
    # TODO: this could be refactored to a method, init and final part could be
    #  refactored into single method
    transformed_data_path = utils.add_text_to_beginning_of_file_path(
        self.preprocessed_data_path, '%s_outlier' % self.transformer.name)
    if os.path.exists(transformed_data_path):
      return pd.read_pickle(transformed_data_path)

    (x_train, y_train), (x_val, y_val), (
      x_test, y_test) = self.get_outlier_detection_datasets()
    # print('train: ', np.unique(y_train, return_counts=True))
    # print('val: ', np.unique(y_val, return_counts=True))
    # print('test: ', np.unique(y_test, return_counts=True))
    x_train_transformed, train_transform_inds = self._perform_transform(x_train)
    x_val_transformed, val_transform_inds = self._perform_transform(x_val)
    x_test_transformed, test_transform_inds = self._perform_transform(x_test)
    sets_tuple = ((x_train_transformed, train_transform_inds),
                  (x_val_transformed, val_transform_inds),
                  (x_test_transformed, test_transform_inds))
    utils.save_pickle(sets_tuple, transformed_data_path)
    return sets_tuple


if __name__ == "__main__":
  from transformations import TransTransformer, Transformer
  import datetime
  import time

  start_time = time.time()
  params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/ztf_v1_bogus_added.pkl'),
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2],
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  transformer = Transformer()
  ztf_outlier_dataset = ZTFOutlierLoader(params, transformer)

  dataset = ztf_outlier_dataset.get_unsplitted_dataset()
  print('dataset: ', np.unique(dataset.data_label, return_counts=True))
  (X_train, y_train), (X_val, y_val), (
    X_test, y_test) = ztf_outlier_dataset.get_outlier_detection_datasets()
  print('train: ', np.unique(y_train, return_counts=True))
  print('val: ', np.unique(y_val, return_counts=True))
  print('test: ', np.unique(y_test, return_counts=True))
  (X_train_trans, y_train_trans), (X_val_trans, y_val_trans), (
  X_test_trans, y_test_trans) = ztf_outlier_dataset.get_transformed_datasets()
  print('train: ', np.unique(y_train_trans, return_counts=True))
  print('val: ', np.unique(y_val_trans, return_counts=True))
  print('test: ', np.unique(y_test_trans, return_counts=True))
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time usage %s: %s" % (transformer.name, str(time_usage)), flush=True)
