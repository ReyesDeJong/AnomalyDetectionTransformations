"""
ZTF stamps outlier loader

safe max data loading float64 (~50GB): (by using float 32 it is reduced to half)
(4500000, 21, 21, 3) == (180*25000, 21, 21, 3)
(500000, 63, 63, 3) == (20*25000, 63, 63, 3)

"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.data_set_generic import Dataset
from parameters import loader_keys, general_keys
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from modules import utils
import matplotlib.pyplot as plt


class ZTFSmallOutlierLoader(ZTFOutlierLoader):

  def __init__(self, params: dict, dataset_name='small_ztf',
      pickles_usage=True):
    self.params = self.get_default_params()
    self.params.update(params)
    self.data_path = params[loader_keys.DATA_PATH]
    self.name = dataset_name
    self.template_save_path = self._get_template_save_path()
    self.crop_size = self.params[loader_keys.CROP_SIZE]
    self.save_pickle = pickles_usage
    self.load_pickle = pickles_usage
    self.random_seed = self.params[general_keys.RANDOM_SEED]

  def get_default_params(self) -> dict:
    default_params = {
      loader_keys.USED_CHANNELS: [0, 1, 2],
      loader_keys.DATA_PATH: None,
      loader_keys.CROP_SIZE: 21,
      general_keys.RANDOM_SEED: 42,
    }
    return default_params

  def _get_template_save_path(self) -> str:
    """get name of final saved file to check if it's been already generated"""
    text_to_add = 'generated_%s/data' % (self.name)
    save_path = utils.add_text_to_beginning_of_file_path(self.data_path,
                                                         text_to_add)
    utils.check_path(os.path.dirname(os.path.abspath(save_path)))
    return save_path

  def get_unsplitted_dataset(self) -> Dataset:
    self.get_outlier_detection_datasets()

  def get_outlier_detection_datasets(self):
    """directly load dataset tuples"""
    (x_train, y_train), (x_val, y_val), (
      x_test, y_test) = pd.read_pickle(self.data_path)
    x_test, y_test = self.shuffle_x_y(x_test, y_test)
    sets_tuple = ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    return sets_tuple

  def plot_image(self, image, save_path=None, show=True, title=None,
      figsize=3):
    image_names = ['template', 'science', 'difference', 'SNR difference']
    n_channels = image.shape[-1]
    fig, axes = plt.subplots(
        1, n_channels, figsize=(figsize * n_channels, figsize + figsize * 0.4))
    for i, ax_i in enumerate(axes):
      if i == 0:
        indx = 1
      elif i == 1:
        indx = 0
      else:
        indx = i
      ax_i.imshow(image[:, :, indx], interpolation='nearest', cmap='gray')
      ax_i.axis('off')
      ax_i.set_title(image_names[indx], fontdict={'fontsize': 15})
    if title:
      fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if save_path:
      data_format = save_path.split('.')[-1]
      fig.savefig(save_path, format=data_format, dpi=600, bbox_inches='tight',
                  pad_inches=0, transparent=True)
    if show:
      plt.show()
    plt.close()


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
