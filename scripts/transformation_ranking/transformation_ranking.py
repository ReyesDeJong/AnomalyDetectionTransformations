"""
Training a model with basic-non-composed transforms, to visualize it feature
 layer and
then calculate FID
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform.transformer_for_ranking import \
  RankingTransformer
from itertools import chain, combinations
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys
import pandas as pd
import numpy as np


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
  # for j in range(5):
  #   pickle_name = 'rank_%i.pkl' % j
  #   for i, trforms_indx_set in enumerate(power_set_clean):
  #     if not os.path.exists(pickle_name):
  #       print(np.array(aux_transformer.transformation_tuples).shape)
  #       trf_to_perform = np.array(aux_transformer.transformation_tuples)[
  #         np.array(trforms_indx_set)]
  #       print(trf_to_perform)
  #       trfer = RankingTransformer()
  #       trfer.set_transformations_to_perform(trf_to_perform.tolist())
  #       model = TransformODSimpleModel(
  #           data_loader, trfer, input_shape=x_train.shape[1:],
  #           results_folder_name=RESULT_PATH)
  #       model.fit(x_train, x_val, epochs=1000, patience=0)
  #       results = model.evaluate_od(x_train, x_test, y_test, data_loader.name,
  #                                   'real', x_val)
  #       results_dict[i] = [trf_to_perform, results]
  #       save_pickle(results_dict[i], 'rank_%i.pkl' % j)
  #     else:
  #       results_dict[i] = pd.read_pickle(pickle_name)
  key_to_get = 'roc_auc'
  # key_to_get = 'pr_auc_anom'
  # key_to_get = 'acc_at_percentil'
  results_dict = pd.read_pickle('rank_3.pkl')
  print(results_dict.keys())
  print(results_dict[0][1]['dirichlet'].keys())
  results_stats = {}

  for key_i in results_dict.keys():
    results_stats[key_i] = [results_dict[key_i][0], []]
    for j in range(5):
      results_dict = pd.read_pickle('rank_%i.pkl' % j)
      results_stats[key_i][1].append(
          results_dict[key_i][1]['dirichlet'][key_to_get])

  print(key_to_get)
  means = []
  stds = []
  trfs_list = []
  for key_i in results_stats.keys():
    mean_i =np.mean(results_stats[key_i][1])
    std_i = np.std(results_stats[key_i][1])
    trf_i = results_stats[key_i][0]
    means.append(mean_i)
    stds.append(std_i)
    trfs_list.append(trf_i)

  sort_idxs = np.argsort(means)
  sort_means = np.array(means)[sort_idxs]
  sort_stds = np.array(stds)[sort_idxs]
  sort_trf_list = np.array(trfs_list)[sort_idxs]

  for i in range(len(sort_trf_list)):
    print(sort_trf_list[i],
            'len %i: %.5f +/- %.5f' % (len(sort_trf_list[i]), sort_means[i], sort_stds[i]))
