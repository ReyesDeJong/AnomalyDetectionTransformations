#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing ZTF database to be saved as a samplesx21x21x3 numpy array in a pickle 

TODO: clean_NaN once cropped
TODO: unit tests
ToDo: instead of cascade implement as pipeline, in order to have single call and definition
ToDo: smart way to shut down nans
@author: asceta
"""
import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from parameters import param_keys
# from modules.data_set_alerce import DatasetAlerce as Dataset
from modules.data_set_generic import Dataset
import numpy as np


# Todo: refactor verbose
# comopse as a pipeline to choose preprocessing steps
class ZTFDataPreprocessor(object):
  """
  Constructor
  """

  def __init__(self, params, verbose=True):
    self.params = params
    self.channels_to_select = params[param_keys.CHANNELS_TO_USE]
    self.number_to_replace_nans = params[param_keys.NANS_TO]
    self.crop_size = params[param_keys.CROP_SIZE]
    self.preprocessing_pipeline = [self.identity]
    self.verbose = verbose

  """
  define your preprocessing strategy here
  """

  def preprocess_dataset(self, dataset: Dataset):
    print('%s' % self._get_string_label_count(dataset.data_label), flush=True)
    for preprocessing_function in self.preprocessing_pipeline:
      dataset = preprocessing_function(dataset)
    self.verbose = False
    return dataset

  def append_to_pipeline(self, method):
    self.preprocessing_pipeline.append(method)
    return self

  def set_pipeline(self, pipeline):
    self.preprocessing_pipeline = pipeline

  def identity(self, dataset: Dataset):
    return dataset

  def check_single_image(self, dataset: Dataset):
    if len(dataset.data_array.shape) == 3:
      dataset.data_array = dataset.data_array[np.newaxis, ...]
    return dataset

  # TODO: erase single image check; adding dummy at begining
  def select_channels(self, dataset: Dataset):
    if len(dataset.data_array.shape) == 3:
      dataset.data_array = dataset.data_array[np.newaxis, ...]
    selected_images_channels = dataset.data_array[
      ..., self.channels_to_select]
    if len(selected_images_channels.shape) == 3:
      selected_images_channels = selected_images_channels[..., np.newaxis]
    dataset.data_array = selected_images_channels
    return dataset

  # TODO: normalize template to avoid replication with by_image
  def normalize_by_sample(self, dataset: Dataset):
    images = dataset.data_array
    images -= np.nanmin(images, axis=(1, 2, 3))[
      ..., np.newaxis, np.newaxis, np.newaxis]
    images = images / np.nanmax(images, axis=(1, 2, 3))[
      ..., np.newaxis, np.newaxis, np.newaxis]
    dataset.data_array = images
    return dataset

  def normalize_by_image(self, dataset: Dataset):
    images = dataset.data_array
    images -= np.nanmin(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
    images = images / np.nanmax(images, axis=(1, 2))[
                      :, np.newaxis, np.newaxis, :]
    dataset.data_array = images
    return dataset

  def nan_to_num(self, dataset: Dataset):
    samples = dataset.data_array
    nans_sample_idx = self._get_nans_samples_idx(samples)
    if self.verbose:
      print('%i samples with NaNs. NaNs replaced with number %s' % (
        len(nans_sample_idx), str(self.number_to_replace_nans)))
    samples[np.isnan(samples)] = self.number_to_replace_nans
    dataset.data_array = samples
    return dataset

  def _check_all_removed(self, remove_name, samples_list, idxs_to_remove):
    if len(samples_list) == len(idxs_to_remove):
      raise OverflowError(
          'All samples have %s, thus batch is empty and cannot be processed' %
          remove_name)

  def _check_misshape_all_removed(self, samples_list, idxs_to_remove):
    self._check_all_removed('MISSHAPE', samples_list, idxs_to_remove)

  def _check_nan_all_removed(self, samples_list, idxs_to_remove):
    self._check_all_removed('NAN', samples_list, idxs_to_remove)

  def _get_misshaped_samples_idx(self, samples):
    miss_shaped_sample_idx = []
    for i in range(len(samples)):
      sample = samples[i]
      if sample.shape[2] != 3 or sample.shape[1] != 63 or sample.shape[0] != 63:
        # print("sample %i of shape %s" % (i, str(sample.shape)))
        miss_shaped_sample_idx.append(i)
    self._check_misshape_all_removed(samples, miss_shaped_sample_idx)
    return miss_shaped_sample_idx

  def clean_misshaped(self, dataset: Dataset):
    samples_clone = list(dataset.data_array[:])
    labels_clone = list(dataset.data_label[:])
    metadata_clone = list(dataset.meta_data[:])
    miss_shaped_sample_idx = self._get_misshaped_samples_idx(samples_clone)
    for index in sorted(miss_shaped_sample_idx, reverse=True):
      samples_clone.pop(index)
      labels_clone.pop(index)
      metadata_clone.pop(index)
    if self.verbose:
      print('%i misshaped samples removed\n%s' % (
        len(miss_shaped_sample_idx),
        self._get_string_label_count(labels_clone)),
            flush=True)
    dataset = Dataset(data_array=samples_clone, data_label=labels_clone,
                      meta_data=metadata_clone,
                      batch_size=dataset.batch_size)
    return dataset

  def _get_nans_samples_idx(self, samples):
    nans_sample_idx = []
    for i in range(len(samples)):
      sample = samples[i]
      if (np.isnan(sample).any()):
        # print("sample %i of shape %s" %(i,str(sample.shape)))
        nans_sample_idx.append(i)
    return nans_sample_idx

  # TODO: refactor; fuse with clean misshaped
  def clean_nans(self, dataset: Dataset):
    samples_clone = list(dataset.data_array[:])
    labels_clone = list(dataset.data_label[:])
    metadata_clone = list(dataset.meta_data[:])
    nans_sample_idx = self._get_nans_samples_idx(samples_clone)
    self._check_nan_all_removed(samples_clone, nans_sample_idx)
    for index in sorted(nans_sample_idx, reverse=True):
      samples_clone.pop(index)
      labels_clone.pop(index)
      metadata_clone.pop(index)
    if self.verbose:
      print('%i samples with NaNs removed\n%s' % (
        len(nans_sample_idx), self._get_string_label_count(labels_clone)),
            flush=True)
    dataset = Dataset(data_array=samples_clone, data_label=labels_clone,
                      batch_size=dataset.batch_size,
                      meta_data=metadata_clone)
    return dataset

  def _get_string_label_count(self, labels):
    class_names = np.array(['AGN', 'SN', 'VS', 'asteroid', 'bogus'])
    label_values, label_counts = np.unique(labels, return_counts=True)
    if len(label_values) != class_names.shape[0]:
      return ""
    count_dict = dict(zip(label_values, label_counts))
    return_str = 'Label count '
    for single_label_value in count_dict.keys():
      return_str += '%s: %i -' % (class_names[single_label_value],
                                  count_dict[single_label_value])
    return return_str

  def crop_at_center(self, dataset: Dataset):
    if self.crop_size is None:
      return dataset
    samples = dataset.data_array
    assert (samples.shape[1] % 2 == self.crop_size % 2)
    center = int((samples.shape[1]) / 2)
    crop_side = int(self.crop_size / 2)
    crop_begin = center - crop_side
    if samples.shape[1] % 2 == 0:
      crop_end = center + crop_side
    elif samples.shape[1] % 2 == 1:
      crop_end = center + crop_side + 1
    # print(center)
    # print(crop_begin, crop_end)
    cropped_samples = samples[:, crop_begin:crop_end, crop_begin:crop_end, :]
    dataset.data_array = cropped_samples
    return dataset
