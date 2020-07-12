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

from modules.transform_selection.fid_modules import fid
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from modules import utils
from models.transformer_od import TransformODModel
from scripts.transformation_selection.histogram_experiments.dict_generator_clf_wrn_fid import ResultDictGeneratorClfWRNFID


class ResultDictGeneratorClfRdmWRNFID(ResultDictGeneratorClfWRNFID):

  def __init__(self, data_loader: HiTSOutlierLoader):
    super().__init__(data_loader)
    self._generator_name = 'ClfRdmWRNFID'
    self._model = self._init_trained_model()

  def _init_trained_model(self):
    (x_train, y_train), (x_val, y_val), (
      x_test, y_test) = self._data_loader.get_outlier_detection_datasets()
    model = self._get_model_instance(x_train)
    return model

  def _get_measures_from_data(self, data):
    activations = self._model.classifier.get_activations(data, batch_size=128)
    mu, sigma = fid. \
      calculate_activation_statistics_from_activation_array(activations)
    return (mu, sigma)


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
    print(x_train.shape)
    csv_gen = ResultDictGeneratorClfRdmWRNFID(data_loader_i)
    csv_gen.get_results_dict(x_train)
    print('')
