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
from parameters import loader_keys
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
from scripts.mutual_info.mutual_info_images_transform_selection. \
  mi_image_HiTS_non_composed_transforms import \
  MIIOverComposedTransformAndHiTSExperiment


class MIIOverComposedTransformAndZTFExperiment(
    MIIOverComposedTransformAndHiTSExperiment):
  def __init__(self, show_plots=True, batch_size=512, n_images=7000,
      windows_size=3, sigma_zero=2.0, transformer=NoCompositionTransformer(),
      dataset_name_and_extra_title='ZTF', random_seed=42):
    super().__init__(
        show_plots, batch_size, n_images, windows_size, sigma_zero, transformer,
        dataset_name_and_extra_title, random_seed)

  def get_data_loader(self):
    ztf_params = {
      loader_keys.DATA_PATH: os.path.join(
          PROJECT_PATH,
          '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
    }
    ztf_loader = ZTFSmallOutlierLoader(ztf_params)
    return ztf_loader



if __name__ == '__main__':
  SHOW_PLOTS = True
  BATCH_SIZE = 512  # 2
  N_IMAGES = 7000  # BATCH_SIZE * 2
  WINDOW_SIZE = 3
  SIGMA_ZERO = 2.0
  experiment = MIIOverComposedTransformAndZTFExperiment(
      show_plots=SHOW_PLOTS,
      batch_size=BATCH_SIZE,
      n_images=N_IMAGES,
      windows_size=WINDOW_SIZE,
      sigma_zero=SIGMA_ZERO)
  experiment.run_experiment()
  print('')