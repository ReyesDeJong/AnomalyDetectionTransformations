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

from modules.geometric_transform.transformer_no_compositions import \
  NoCompositionTransformer
from modules.transform_selection.fid_modules import fid
import numpy as np
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules import utils


class ResultDictGeneratorRawFID(object):

  def __init__(self, data_loader: HiTSOutlierLoader):
    self._generator_name = 'RawFID'
    self._data_loader = data_loader
    self._transformer = NoCompositionTransformer()
    self._dict_trf_idx_name = self.create_transfor_indx_name_dict()
    self._results_dict_trf_name_distance = self.create_transfor_name_idx_dict()

  def change_dataloader(self, data_loader: HiTSOutlierLoader):
    self._data_loader = data_loader

  def create_transfor_indx_name_dict(self) -> dict:
    trf_dict = {}
    transformer = self._transformer
    tuple_names = ['flip', 'shiftX', 'shiftY', 'rot', 'gauss',
                   'laplace', 'none']
    for transform_idx in range(transformer.n_transforms):
      tuple_trf = transformer.transformation_tuples[transform_idx]
      tuple_selected_idx = np.argwhere(np.array(tuple_trf) != 0).squeeze()
      if tuple_selected_idx.tolist() == []:
        tuple_selected_idx = -1
      selected_tuple_name = tuple_names[tuple_selected_idx]
      if selected_tuple_name in ['shiftX', 'rot', 'shiftY']:
        trf_name = "%s_%s" % (
          selected_tuple_name, tuple_trf[tuple_selected_idx])
      else:
        trf_name = "%s" % (selected_tuple_name)
      trf_dict[transform_idx] = trf_name
    return trf_dict

  def create_transfor_name_idx_dict(self) -> dict:
    trf_dict = self.create_transfor_indx_name_dict()
    trf_dict = {v: k for k, v in trf_dict.items()}
    return trf_dict

  def _get_measures_from_data(self, data):
    mu, sigma = fid. \
      calculate_activation_statistics_from_activation_array(data)
    return (mu, sigma)

  def _get_distance_original_trf(self, measure_original, measure_trf):
    fid_value = fid.calculate_frechet_distance(
        measure_original[0], measure_original[1], measure_trf[0],
        measure_trf[1])
    return fid_value

  def _fill_results_dict(self, features_original,
      features_to_transform=None):
    if features_to_transform == None:
      features_to_transform = features_original
    measure_original = self._get_measures_from_data(features_original)
    n_transforms = self._transformer.n_transforms
    for transform_idx in range(n_transforms):
      features_trf = \
        self._transformer.apply_transforms(features_to_transform,
                                           [transform_idx])[0]
      measure_trf = self._get_measures_from_data(features_trf)
      dist_value = self._get_distance_original_trf(measure_original,
                                                   measure_trf)
      trf_name = self._dict_trf_idx_name[transform_idx]
      self._results_dict_trf_name_distance[trf_name] = dist_value

  def get_results_dict(self, features_original,
      features_to_transform=None, save_to_folder=True, exp_name=None,
      save_folder_path=None):
    self._fill_results_dict(features_original, features_to_transform)
    if save_to_folder:
      if exp_name is None:
        exp_name = self._generator_name
      file_name = '%s_%s.pkl' % (exp_name, self._data_loader.name)
      if save_folder_path is None:
        save_folder_path = os.path.join(
            PROJECT_PATH, 'scripts', 'transformation_selection',
            'histogram_experiments', 'results')
      save_path = os.path.join(save_folder_path, file_name)
      utils.check_paths(save_folder_path)
      utils.save_pickle(self._results_dict_trf_name_distance, save_path)
    return self._results_dict_trf_name_distance


if __name__ == "__main__":
  from parameters import loader_keys, general_keys

  utils.init_gpu_soft_growth()
  # data loaders
  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 100000,
    loader_keys.TEST_PERCENTAGE: 0.0,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
    loader_keys.USED_CHANNELS: [2],#[0, 1, 2, 3],  #
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  hits_loader = HiTSOutlierLoader(hits_params, pickles_usage=False)
  data_loader = hits_loader
  x_train = data_loader.get_outlier_detection_datasets()[0][0]
  csv_gen = ResultDictGeneratorRawFID(hits_loader)
  print(csv_gen.get_results_dict(x_train[:2000]))
