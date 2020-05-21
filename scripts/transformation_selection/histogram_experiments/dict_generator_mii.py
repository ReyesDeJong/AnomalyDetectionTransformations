"""
Training a model with basic-non-composed transforms, to visualize it feature
 layer and
then calculate FID
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from modules import utils
from scripts.transformation_selection.histogram_experiments.dict_generator_raw_fid import \
  ResultDictGeneratorRawFID
from modules.transform_selection.mutual_info.mi_image_calculator import \
  MIImageCalculator
from modules.info_metrics.information_estimator_by_batch import \
  InformationEstimatorByBatch
from modules import score_functions


class ResultDictGeneratorMII(ResultDictGeneratorRawFID):

  def __init__(self, data_loader: HiTSOutlierLoader, batch_size=512,
      windows_size=3, sigma_zero=2.0, norm_images=True):
    super().__init__(data_loader)
    self._generator_name = 'MIIEntropyNorm%i' % int(norm_images * 1)
    self._norm_patches = norm_images
    self.mi_estimator = InformationEstimatorByBatch(sigma_zero, batch_size)
    self.mi_image_calculator = MIImageCalculator(
        information_estimator=self.mi_estimator, window_size=windows_size)

  def _get_measures_from_data(self, data):
    return data

  def _get_distance_original_trf(self, measure_original, measure_trf):
    mii = self.mi_image_calculator.mi_images_mean(
        measure_original, measure_trf, normalize_patches=self._norm_patches)
    print(mii.shape)
    comparison_value = score_functions.get_entropy(mii[None, ...])[
      0]
    return comparison_value


if __name__ == "__main__":
  from parameters import loader_keys, general_keys

  utils.init_gpu_soft_growth()
  # data loaders
  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    # loader_keys.N_SAMPLES_BY_CLASS: 100000,
    loader_keys.N_SAMPLES_BY_CLASS: 1000,
    loader_keys.TEST_PERCENTAGE: 0.0,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
    loader_keys.USED_CHANNELS: [2],  # [0, 1, 2, 3],  #
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  hits_loader_1c = HiTSOutlierLoader(hits_params, pickles_usage=False)
  hits_params.update({loader_keys.USED_CHANNELS: [0, 1, 2, 3]})
  hits_loader_4c = HiTSOutlierLoader(hits_params, pickles_usage=False)
  ztf_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH,
        '../datasets/ALeRCE_data/converted_pancho_septiembre.pkl'),
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
    loader_keys.USED_CHANNELS: [0, 1, 2],
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  ztf_loader_3c = ZTFOutlierLoader(ztf_params, pickles_usage=False)
  ztf_params.update({loader_keys.USED_CHANNELS: [2]})
  ztf_loader_1c = ZTFOutlierLoader(ztf_params, pickles_usage=False)

  data_loaders = [hits_loader_1c, hits_loader_4c,
                  ztf_loader_1c, ztf_loader_3c]

  for data_loader_i in data_loaders:
    x_train = data_loader_i.get_outlier_detection_datasets()[0][0]
    csv_gen = ResultDictGeneratorMII(data_loader_i)
    csv_gen.get_results_dict(x_train)
    print('')
