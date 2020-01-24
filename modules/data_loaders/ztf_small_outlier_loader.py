"""
ZTF stamps outlier loader

safe max data loading float64 (~50GB): (by using float 32 it is reduced to half)
(4500000, 21, 21, 3) == (180*25000, 21, 21, 3)
(500000, 63, 63, 3) == (20*25000, 63, 63, 3)

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
from modules.data_loaders.frame_to_input import FrameToInput
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from modules import utils

class ZTFSmallOutlierLoader(ZTFOutlierLoader):

  def __init__(self, params: dict, dataset_name='small_ztf'):
    self.data_path = params[loader_keys.DATA_PATH]
    self.name = dataset_name
    self.template_save_path = self._get_template_save_path()
    self.crop_size = 21

  def _get_template_save_path(self) -> str:
    """get name of final saved file to check if it's been already generated"""
    text_to_add = 'generated_%s/data' % (self.name)
    save_path = utils.add_text_to_beginning_of_file_path(self.data_path, text_to_add)
    utils.check_path(os.path.dirname(os.path.abspath(save_path)))
    return save_path

  def get_unsplitted_dataset(self) -> Dataset:
    self.get_outlier_detection_datasets()

  def get_outlier_detection_datasets(self):
    """directly load dataset tuples"""
    return pd.read_pickle(self.data_path)

if __name__ == "__main__":
  from modules.geometric_transform.transformations_tf import Transformer
  import datetime
  import time

  start_time = time.time()
  params = {
    loader_keys.DATA_PATH: '/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl',
  }
  transformer = Transformer()
  ztf_outlier_dataset = ZTFSmallOutlierLoader(params)

  (X_train, y_train), (X_val, y_val), (
    X_test, y_test) = ztf_outlier_dataset.get_outlier_detection_datasets()
  print('train: ', np.unique(y_train, return_counts=True))
  print('val: ', np.unique(y_val, return_counts=True))
  print('test: ', np.unique(y_test, return_counts=True))
  (X_train_trans, y_train_trans), (X_val_trans, y_val_trans), (
    X_test_trans, y_test_trans) = ztf_outlier_dataset.get_transformed_datasets(
      transformer)
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time usage %s: %s" % (transformer.name, str(time_usage)), flush=True)
  print('train: ', np.unique(y_train_trans, return_counts=True))
  print('val: ', np.unique(y_val_trans, return_counts=True))
  print('test: ', np.unique(y_test_trans, return_counts=True))

