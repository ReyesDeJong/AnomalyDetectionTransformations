"""
MII for every transformatioon on toy data
experiments on patch norm or MII individual norm
MII no patch norm and no individual norm gives better results
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)
from modules.transform_selection.mutual_info.mi_image_calculator import \
  MIImageCalculator
from modules.info_metrics.information_estimator_by_batch import \
  InformationEstimatorByBatch
from modules.geometric_transform.transformer_no_compositions import \
  NoCompositionTransformer
import tensorflow as tf
from parameters import loader_keys
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader


if __name__ == '__main__':
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  SHOW_PLOTS = True
  BATCH_SIZE = 512
  WINDOW_SIZE = 3
  SIGMA_ZERO = 2.0

  ztf_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH,
        '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
  }
  ztf_loader = ZTFSmallOutlierLoader(ztf_params)
  data_loader = ztf_loader
  (x_train, y_train), (
    x_val, y_val), _ = data_loader.get_outlier_detection_datasets()
  mi_estimator = InformationEstimatorByBatch(SIGMA_ZERO, BATCH_SIZE)
  mi_image_calculator = MIImageCalculator(information_estimator=mi_estimator,
                                          window_size=WINDOW_SIZE)
  transformer = NoCompositionTransformer()
  images = x_train
  mii_every_transform = mi_image_calculator.mii_for_transformations(
      images, transformer)
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=False, extra_title_text='ZTF')
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=True, extra_title_text='ZTF')
  mii_every_transform = mi_image_calculator.mii_for_transformations(
      images, transformer, normalize_patches=True)
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=False,
                                    extra_title_text='ZTF normed patches')
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=True,
                                    extra_title_text='ZTF normed patches')
  print('')