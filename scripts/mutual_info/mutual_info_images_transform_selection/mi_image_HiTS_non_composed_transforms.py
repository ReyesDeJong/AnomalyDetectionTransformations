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
from modules.transform_selection.mutual_info.mi_image_calculator import \
  MIImageCalculator
from modules.info_metrics.information_estimator_by_batch import \
  InformationEstimatorByBatch
from modules.geometric_transform.transformer_no_compositions import \
  NoCompositionTransformer
import tensorflow as tf
from parameters import loader_keys, general_keys
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader


if __name__ == '__main__':
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  SHOW_PLOTS = True
  BATCH_SIZE = 512
  WINDOW_SIZE = 3
  SIGMA_ZERO = 2.0

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
  data_loader = hits_loader
  (x_train, y_train), (
    x_val, y_val), _ = data_loader.get_outlier_detection_datasets()
  mi_estimator = InformationEstimatorByBatch(SIGMA_ZERO, BATCH_SIZE)
  mi_image_calculator = MIImageCalculator(information_estimator=mi_estimator,
                                          window_size=WINDOW_SIZE)
  transformer = NoCompositionTransformer()
  images = x_train
  mii_every_transform = mi_image_calculator.mii_for_transformations(
      images, transformer)
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=False, extra_title_text='HITS')
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=True, extra_title_text='HITS')
  mii_every_transform = mi_image_calculator.mii_for_transformations(
      images, transformer, normalize_patches=True)
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=False,
                                    extra_title_text='HITS normed patches')
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=True,
                                    extra_title_text='HITS normed patches')
  print('')