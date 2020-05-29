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

from modules.geometric_transform.transformer_for_ranking import \
  RankingTransformer
from modules.data_loaders.artificial_dataset_factory import CirclesFactory
from itertools import chain, combinations
from models.transformer_od_simple_net import TransformODSimpleModel
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys


# class ResultDictGeneratorRawFID(object):
#
#   def __init__(self, data_loader: HiTSOutlierLoader):


def get_powerset(iterable):
  "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  s = list(iterable)
  return list(
      chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


if __name__ == "__main__":
  RESULT_PATH = os.path.join(PROJECT_PATH, 'results', 'Trf_Rank')
  # cicles_factory = CirclesFactory()
  # samples = cicles_factory.get_final_dataset(n_images=10)
  aux_transformer = RankingTransformer()
  n_tuple_array = list(range(aux_transformer.n_transforms))
  power_set = get_powerset(n_tuple_array)
  power_set_clean = [x for x in power_set if len(x) > 1 and 0 in x]

  print(power_set_clean)
  print(len(power_set_clean))
  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  data_loader = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader.get_outlier_detection_datasets()
  results_dict = {}
  for trforms_indx_set in power_set_clean[:2]:
    trf_to_perform = aux_transformer.transformation_tuples[
      power_set_clean[trforms_indx_set]]
    trfer = RankingTransformer()
    trfer.set_transformations_to_perform(trf_to_perform)
    model = TransformODSimpleModel(
        data_loader, trfer, input_shape=x_train.shape[1:],
        results_folder_name=RESULT_PATH)
    model.fit(x_train, x_val, epochs=1e100)
    results = model.evaluate_od(x_train, x_test, y_test, data_loader.name,
                                'real', x_val)
    results_dict[trf_to_perform] = results
