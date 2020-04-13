"""
MII for every transformatioon on toy data
experiments on patch norm or MII individual norm
MII no patch norm and no individual norm gives better results
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
from modules.geometric_transform.transformer_no_compositions import \
  NoCompositionTransformer
from parameters import loader_keys, general_keys
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from scripts.mutual_info.mutual_info_images_transform_selection. \
  mi_image_over_toy_data_non_composed_transforms import \
  MIIOverComposedTransformAndToyDataExperiment


class MIIOverComposedTransformAndHiTSExperiment(
    MIIOverComposedTransformAndToyDataExperiment):
  def __init__(self, show_plots=True, batch_size=512, n_images=7000,
      windows_size=3, sigma_zero=2.0, transformer=NoCompositionTransformer(),
      dataset_name_and_extra_title='HiTS', random_seed=42):
    super().__init__(
        show_plots, batch_size, n_images, windows_size, sigma_zero, transformer,
        dataset_name_and_extra_title)
    self.random_seed = random_seed

  def get_data_loader(self):
    hits_params = {
      loader_keys.DATA_PATH: os.path.join(
          PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
      loader_keys.N_SAMPLES_BY_CLASS: 10000,
      loader_keys.TEST_PERCENTAGE: 0.2,
      loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
      loader_keys.USED_CHANNELS: [0, 1, 2, 3],  # [2],#
      loader_keys.CROP_SIZE: 21,
      general_keys.RANDOM_SEED: 42,
      loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    hits_loader = HiTSOutlierLoader(hits_params)
    return hits_loader

  def sub_sample_images(self, images):
    idxs = np.arange(len(images))
    selected_idxs = np.random.RandomState(self.random_seed).choice(
        idxs, self.n_images, replace=False)
    return images[selected_idxs]

  def get_images(self):
    data_loader = self.get_data_loader()
    (x_train, y_train), (
      x_val, y_val), _ = data_loader.get_outlier_detection_datasets()
    images = x_train
    images = self.sub_sample_images(images)
    return images

  def run_experiment(self):
    self.set_tf_gpus()
    images = self.get_images()
    mii_every_transform = self.mi_image_calculator.mii_for_transformations(
        images, self.transformer)
    mii_every_transform.plot_mii_dict(
        plot_show=self.show_plots, norm_mii=False,
        extra_title_text=self.dataset_name_and_extra_title)
    mii_every_transform.plot_mii_dict(
        plot_show=self.show_plots, norm_mii=True,
        extra_title_text=self.dataset_name_and_extra_title)
    mii_every_transform = self.mi_image_calculator.mii_for_transformations(
        images, self.transformer, normalize_patches=True)
    mii_every_transform.plot_mii_dict(
        plot_show=self.show_plots, norm_mii=False,
        extra_title_text='normed patches ' + self.dataset_name_and_extra_title)
    mii_every_transform.plot_mii_dict(
        plot_show=self.show_plots, norm_mii=True,
        extra_title_text='normed patches ' + self.dataset_name_and_extra_title)


if __name__ == '__main__':
  SHOW_PLOTS = True
  BATCH_SIZE = 512  # 2
  N_IMAGES = 7000  # BATCH_SIZE * 2
  WINDOW_SIZE = 3
  SIGMA_ZERO = 2.0
  experiment = MIIOverComposedTransformAndHiTSExperiment(
      show_plots=SHOW_PLOTS,
      batch_size=BATCH_SIZE,
      n_images=N_IMAGES,
      windows_size=WINDOW_SIZE,
      sigma_zero=SIGMA_ZERO)
  experiment.run_experiment()
  print('')
