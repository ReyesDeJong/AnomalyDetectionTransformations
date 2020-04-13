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


class MIIOverComposedTransformAndToyDataExperiment(object):
  def __init__(self, show_plots=True, batch_size=512, n_images=7000,
      windows_size=3, sigma_zero=2.0, transformer=NoCompositionTransformer(),
      dataset_name_and_extra_title='Toy+noise'):
    self.show_plots = show_plots
    # self.batch_size = batch_size
    self.n_images = n_images
    # self.windows_size = windows_size
    # self.sigma_zero = sigma_zero
    self.dataset_name_and_extra_title = dataset_name_and_extra_title
    self.mi_estimator = InformationEstimatorByBatch(sigma_zero, batch_size)
    self.mi_image_calculator = MIImageCalculator(
        information_estimator=self.mi_estimator, window_size=windows_size)
    self.transformer = transformer

  def set_tf_gpus(self):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

  def get_images(self):
    circle_factory = CirclesFactory()
    images = circle_factory.get_final_dataset(self.n_images)
    return images

  def run_experiment(self):
    self.set_tf_gpus()
    images = self.get_images()
    mii_every_transform = self.mi_image_calculator.mii_for_transformations(
        images, self.transformer)
    mii_every_transform.plot_mii_dict(
        plot_show=SHOW_PLOTS, norm_mii=False,
        extra_title_text=self.dataset_name_and_extra_title)
    mii_every_transform.plot_mii_dict(
        plot_show=SHOW_PLOTS, norm_mii=True,
        extra_title_text=self.dataset_name_and_extra_title)
    # mii_every_transform = mi_image_calculator.mii_for_transformations(
    #     images, transformer, normalize_patches=True)
    # mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=False,
    #                                   extra_title_text='normed patches')
    # mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=True,
    #                                   extra_title_text='normed patches')


if __name__ == '__main__':
  SHOW_PLOTS = True
  BATCH_SIZE = 512  # 2
  N_IMAGES = BATCH_SIZE * 4  # 7000  # BATCH_SIZE * 2
  WINDOW_SIZE = 3
  SIGMA_ZERO = 2.0
  experiment = MIIOverComposedTransformAndToyDataExperiment(
      show_plots=SHOW_PLOTS,
      batch_size=BATCH_SIZE,
      n_images=N_IMAGES,
      windows_size=WINDOW_SIZE,
      sigma_zero=SIGMA_ZERO)
  experiment.run_experiment()
  print('')
