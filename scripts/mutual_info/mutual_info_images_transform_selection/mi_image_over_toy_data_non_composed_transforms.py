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
from modules.transform_selection.artificial_dataset_factory import \
  CirclesFactory
from modules.transform_selection.mutual_info.mi_image_calculator import \
  MIImageCalculator
from modules.info_metrics.information_estimator_by_batch import \
  InformationEstimatorByBatch
from modules.geometric_transform.transformer_no_compositions import \
  NoCompositionTransformer
import tensorflow as tf


if __name__ == '__main__':
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  SHOW_PLOTS = True
  BATCH_SIZE = 512  # 2
  N_IMAGES = BATCH_SIZE * 4  # 7000  # BATCH_SIZE * 2
  WINDOW_SIZE = 3
  SIGMA_ZERO = 2.0

  circle_factory = CirclesFactory()
  mi_estimator = InformationEstimatorByBatch(SIGMA_ZERO, BATCH_SIZE)
  mi_image_calculator = MIImageCalculator(information_estimator=mi_estimator,
                                          window_size=WINDOW_SIZE)
  transformer = NoCompositionTransformer()
  images = circle_factory.get_final_dataset(N_IMAGES)
  mii_every_transform = mi_image_calculator.mii_for_transformations(
      images, transformer)
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=False)
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=True)
  # mii_every_transform = mi_image_calculator.mii_for_transformations(
  #     images, transformer, normalize_patches=True)
  # mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=False,
  #                                   extra_title_text='normed patches')
  # mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=True,
  #                                   extra_title_text='normed patches')
  print('')