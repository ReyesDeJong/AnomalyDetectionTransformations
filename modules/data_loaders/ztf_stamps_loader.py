from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.data_loaders.hits_loader import HiTSLoader
from modules.data_set_generic import Dataset
from modules.data_splitters.data_splitter_n_samples import \
  DatasetDividerInt
from parameters import general_keys, param_keys
from modules.data_loaders.ztf_preprocessor import ZTFDataPreprocessor

"""
ztf stamps data loader
"""


# TODO: evaluate if it's good idea to pass params and use batchsize in
# dataset_generic
class ZTFLoader(HiTSLoader):
  """
  Constructor
  """

  def __init__(self, params: dict):
    self.path = params[param_keys.DATA_PATH_TRAIN]
    self.batch_size = params[param_keys.BATCH_SIZE]
    self.data_splitter = DatasetDividerInt(
        test_size=params[param_keys.TEST_SIZE],
        validation_size=params[param_keys.VAL_SIZE])
    self.channel_to_get = params[param_keys.CHANNELS_TO_USE]
    self.dataset_preprocessor = self._create_dataset_preprocessor(params)

  def _dict_to_dataset(self, data_dict):
    dataset = Dataset(data_array=data_dict[general_keys.IMAGES],
                      data_label=data_dict[general_keys.LABELS],
                      batch_size=self.batch_size)
    return dataset

  def _get_data(self, path):
    data_dict = self._load_data(path)
    return data_dict

  def _create_dataset_preprocessor(self, params):
    dataset_preprocessor = ZTFDataPreprocessor(params)
    dataset_preprocessor.set_pipeline(
        [dataset_preprocessor.check_single_image,
         dataset_preprocessor.clean_misshaped,
         dataset_preprocessor.select_channels,
         dataset_preprocessor.clean_nans,
         dataset_preprocessor.normalize_by_sample])
    return dataset_preprocessor

  def _get_preprocessed_dataset(self, data_dict):
    dataset = self._dict_to_dataset(data_dict)
    dataset_prepocessed = self.dataset_preprocessor.preprocess_dataset(dataset)
    return dataset_prepocessed

  def get_dataset_preprocessor(self):
    return self.dataset_preprocessor

  def get_datasets(self) -> dict:
    data_dict = self._get_data(self.path)
    dataset = self._get_preprocessed_dataset(data_dict)
    datasets_dict = self._init_datasets_dict()
    self.data_splitter.set_dataset_obj(dataset)
    train_dataset, test_dataset, val_dataset = \
      self.data_splitter.get_train_test_val_set_objs()
    datasets_dict[general_keys.TRAIN] = train_dataset
    datasets_dict[general_keys.TEST] = test_dataset
    datasets_dict[general_keys.VALIDATION] = val_dataset
    return datasets_dict

  def get_single_dataset(self) -> Dataset:
    data_dict = self._get_data(self.path)
    dataset = self._get_preprocessed_dataset(data_dict)
    return dataset


if __name__ == "__main__":
  parameters = {
    param_keys.DATA_PATH_TRAIN: os.path.join(PROJECT_PATH, '..', 'datasets',
                                             'ZTF', 'stamp_classifier',
                                             'ztf_dataset.pkl'),
    param_keys.TEST_SIZE: 100,
    param_keys.VAL_SIZE: 50,
    param_keys.BATCH_SIZE: 50,
    param_keys.CHANNELS_TO_USE: [0, 1, 2],
    param_keys.NANS_TO: 1,
    param_keys.INPUT_IMAGE_SIZE: 63,
  }
  data_loader = ZTFLoader(parameters)
  datasets_dict = data_loader.get_datasets()
  print('train %s' % str(datasets_dict[general_keys.TRAIN].data_array.shape))
  print('test %s' % str(datasets_dict[general_keys.TEST].data_array.shape))
  print('val %s' % str(datasets_dict[general_keys.VALIDATION].data_array.shape))

  # replication test
  train_dataset = datasets_dict[general_keys.TRAIN]
  print('train samples %s, train SNe %i' % (str(train_dataset.data_array.shape),
                                            int(np.sum(
                                                train_dataset.data_label))))
  train_dataset.balance_data_by_replication_2_classes()
  print('train samples %s, train SNe %i' % (str(train_dataset.data_array.shape),
                                            int(np.sum(
                                                train_dataset.data_label))))
