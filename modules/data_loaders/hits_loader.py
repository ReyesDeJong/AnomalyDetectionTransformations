from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pkl
import sys

import numpy as np

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.data_set_generic import Dataset
from modules.data_splitter import DatasetDivider
from parameters import general_keys, param_keys

"""
hist2013 data loader
"""


# TODO: evaluate if it's good idea to pass params and use batchsize in
# dataset_generic
class HiTSLoader(object):
  """
  Constructor
  """

  def __init__(self, params: dict, label_value=1, first_n_samples_by_class=125000, test_size=0.12, validation_size=0.08):
    self.path = params[param_keys.DATA_PATH_TRAIN]
    self.batch_size = params[param_keys.BATCH_SIZE]
    self.data_splitter = DatasetDivider(data_set_obj=None, test_size=test_size, validation_size=validation_size)
    self.first_n_samples_by_class = first_n_samples_by_class
    self.label_value = label_value
    self.channel_to_get = 2

  def _init_datasets_dict(self):
    datasets_dict = {
      general_keys.TRAIN: None,
      general_keys.VALIDATION: None,
      general_keys.TEST: None
    }
    return datasets_dict

  def _load_file(self, path):
    infile = open(path, 'rb')
    data = pkl.load(infile)
    return data

  def _get_first_n_samples_by_label(self, data_dict, n_samples, label_value):
    images = data_dict[general_keys.IMAGES]
    labels = data_dict[general_keys.LABELS]
    # print(labels.shape)
    label_value_idxs = np.where(labels == label_value)[0]
    # print(label_value_idxs.shape)
    np.random.shuffle(label_value_idxs)
    label_idxs_to_get = label_value_idxs[:n_samples]
    data_dict[general_keys.IMAGES] = images[label_idxs_to_get]
    data_dict[general_keys.LABELS] = labels[label_idxs_to_get]
    return data_dict

  def normalize_images(self, images):
    # normilize 0-1
    for image_index in range(images.shape[0]):
      image = images[image_index]
      image -= image.min()
      image = image / image.max()
      images[image_index] = image
    return images

  def _preprocess_data(self, data_dict, first_n_samples, label_value):
    data_dict = self._get_first_n_samples_by_label(
        data_dict, n_samples=first_n_samples, label_value=label_value)
    # get difference image
    # Todo: code as param to get channel
    selected_image_channels = data_dict[general_keys.IMAGES][
      ..., self.channel_to_get]
    if len(selected_image_channels.shape) == 3:
      selected_image_channels = selected_image_channels[..., np.newaxis]
    data_dict[general_keys.IMAGES] = selected_image_channels
    # normalice images 0-1
    data_dict[general_keys.IMAGES] = self.normalize_images(
        data_dict[general_keys.IMAGES])
    return data_dict

  def get_datasets(self) -> dict:
    data_dict = self._load_file(self.path)
    data_dict = self._preprocess_data(data_dict, self.first_n_samples_by_class,
                                      self.label_value)
    dataset = Dataset(data_array=data_dict[general_keys.IMAGES],
                      data_label=data_dict[general_keys.LABELS],
                      batch_size=self.batch_size)
    datasets_dict = self._init_datasets_dict()
    self.data_splitter.set_dataset_obj(dataset)
    train_dataset, test_dataset, val_dataset = \
      self.data_splitter.get_train_test_val_set_objs()
    datasets_dict[general_keys.TRAIN] = train_dataset
    datasets_dict[general_keys.TEST] = test_dataset
    datasets_dict[general_keys.VALIDATION] = val_dataset
    return datasets_dict

  def change_labels_values(self, datasets_dict: dict,
      new_labels_value: int) -> dict:
    for set_name in datasets_dict.keys():
      datasets_dict[set_name].data_label = np.ones_like(
          datasets_dict[set_name].data_label) * new_labels_value
    return datasets_dict

  def _concatenate_data_dicts(self, data_dicts_list: list) -> dict:
    merged_data_dict = {}
    for key in data_dicts_list[0].keys():
      merged_data_dict[key] = np.concatenate(
          [data_dict[key] for data_dict in data_dicts_list])
    return merged_data_dict

  def load_data(self):
    """get first n_samples_by_class data from each hits class"""
    data_dict = self._load_file(self.path)
    # get dataset by label
    unique_label_values = np.unique(data_dict[general_keys.LABELS])
    list_of_data_dicts_by_label = []
    for label_value in unique_label_values:
      specific_label_data_dict = self._preprocess_data(
          data_dict.copy(), self.first_n_samples_by_class, label_value)
      list_of_data_dicts_by_label.append(specific_label_data_dict)
    merged_data_dict = self._concatenate_data_dicts(
        list_of_data_dicts_by_label)
    #split data into train-test
    dataset = Dataset(data_array=merged_data_dict[general_keys.IMAGES],
                      data_label=merged_data_dict[general_keys.LABELS],
                      batch_size=self.batch_size)

    self.data_splitter.set_dataset_obj(dataset)
    train_dataset, test_dataset, val_dataset = \
      self.data_splitter.get_train_test_val_set_objs()
    # print(np.unique(train_dataset.data_label,
    #                 return_counts=True))

    return (train_dataset.data_array, train_dataset.data_label), \
           (test_dataset.data_array, test_dataset.data_label)


if __name__ == "__main__":
  params = {
    param_keys.DATA_PATH_TRAIN: os.path.join(PROJECT_PATH, '..', 'datasets',
                                             'HiTS2013_300k_samples.pkl'),
    param_keys.BATCH_SIZE: 50
  }
  data_loader = HiTSLoader(params)
  datasets_dict = data_loader.get_datasets()
  print('train %s' % str(datasets_dict[general_keys.TRAIN].data_array.shape))
  print('test %s' % str(datasets_dict[general_keys.TEST].data_array.shape))
  print('val %s' % str(datasets_dict[general_keys.VALIDATION].data_array.shape))

  (X_train, y_train), (X_test, y_test) = data_loader.load_data()
  print('All data train shape %s, label mean %.1f' % (str(X_train.shape),
        np.mean(y_train)))
  print('All data test shape %s, label mean %.1f' % (str(X_test.shape),
        np.mean(y_test)))
