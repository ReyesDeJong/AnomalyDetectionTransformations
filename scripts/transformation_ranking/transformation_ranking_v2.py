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
from models.transformer_od_simple_net import TransformODSimpleModel
from models.transformer_od import TransformODModel
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
from modules.geometric_transform.transformations_tf import Transformer, \
  TransTransformer
from itertools import chain, combinations
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys
import numpy as np
from modules.utils import save_pickle


def get_powerset(iterable):
  "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  s = list(iterable)
  return list(
      chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

def main():
  RESULT_PATH = os.path.join(PROJECT_PATH, 'results', 'Trf_Rank')
  N_RUNS = 10

  aux_transformer = RankingTransformer()
  trf_72 = Transformer()
  trf_9 = TransTransformer()
  n_tuple_array = list(range(aux_transformer.n_transforms))
  power_set = get_powerset(n_tuple_array)
  power_set_clean = [x for x in power_set if len(x) > 1 and 0 in x]
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
  hits_loader = HiTSOutlierLoader(hits_params)
  ztf_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH,
        '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
  }
  ztf_loader = ZTFSmallOutlierLoader(ztf_params)

  data_loaders = (hits_loader, ztf_loader)

  for loader_i in data_loaders:
    (x_train, y_train), (x_val, y_val), (
      x_test, y_test) = loader_i.get_outlier_detection_datasets()
    pickle_name = 'rank_%s.pkl' % loader_i.name
    result_all_runs = {}
    for run_i in range(N_RUNS):
      indexes_for_power_set = list(range(len(power_set_clean))) + [-1, -2, -3,
                                                                   -4]
      # indexes_for_power_set = list(range(len(power_set_clean)))[:3] + [-1]
      result_each_trf = {}
      for power_set_idx in indexes_for_power_set:
        if power_set_idx == -1:
          model = TransformODModel(
              loader_i, trf_72, input_shape=x_train.shape[1:],
              results_folder_name=RESULT_PATH)
          trf_to_perform = np.array(trf_72.transformation_tuples)
        elif power_set_idx == -2:
          model = TransformODModel(
              loader_i, trf_9, input_shape=x_train.shape[1:],
              results_folder_name=RESULT_PATH)
          trf_to_perform = np.array(trf_9.transformation_tuples)
        elif power_set_idx == -3:
          model = TransformODSimpleModel(
              loader_i, trf_72, input_shape=x_train.shape[1:],
              results_folder_name=RESULT_PATH)
          trf_to_perform = np.array(trf_72.transformation_tuples)
        elif power_set_idx == -4:
          model = TransformODSimpleModel(
              loader_i, trf_9, input_shape=x_train.shape[1:],
              results_folder_name=RESULT_PATH)
          trf_to_perform = np.array(trf_9.transformation_tuples)
        else:
          trforms_indx_set = power_set_clean[power_set_idx]
          trf_to_perform = np.array(aux_transformer.transformation_tuples)[
            np.array(trforms_indx_set)]
          print(trf_to_perform)
          trfer = RankingTransformer()
          trfer.set_transformations_to_perform(trf_to_perform.tolist())
          model = TransformODSimpleModel(
              loader_i, trfer, input_shape=x_train.shape[1:],
              results_folder_name=RESULT_PATH)
        model.fit(x_train, x_val, epochs=1000, patience=0)
        results = model.evaluate_od(x_train, x_test, y_test, loader_i.name,
                                    'real', x_val)
        result_each_trf[power_set_idx] = [trf_to_perform, results]
      result_all_runs[run_i] = result_each_trf
    save_pickle(result_all_runs, pickle_name)

if __name__ == "__main__":
  print('')
  # key_to_get = 'roc_auc'
  # # key_to_get = 'pr_auc_anom'
  # # key_to_get = 'acc_at_percentil'
  # results_dict = pd.read_pickle('rank_3.pkl')
  # print(results_dict.keys())
  # print(results_dict[0][1]['dirichlet'].keys())
  # results_stats = {}
  #
  # for key_i in results_dict.keys():
  #   results_stats[key_i] = [results_dict[key_i][0], []]
  #   for j in range(5):
  #     results_dict = pd.read_pickle('rank_%i.pkl' % j)
  #     results_stats[key_i][1].append(
  #         results_dict[key_i][1]['dirichlet'][key_to_get])
  #
  # print(key_to_get)
  # means = []
  # stds = []
  # trfs_list = []
  # for key_i in results_stats.keys():
  #   mean_i =np.mean(results_stats[key_i][1])
  #   std_i = np.std(results_stats[key_i][1])
  #   trf_i = results_stats[key_i][0]
  #   means.append(mean_i)
  #   stds.append(std_i)
  #   trfs_list.append(trf_i)
  #
  # sort_idxs = np.argsort(means)
  # sort_means = np.array(means)[sort_idxs]
  # sort_stds = np.array(stds)[sort_idxs]
  # sort_trf_list = np.array(trfs_list)[sort_idxs]
  #
  # for i in range(len(sort_trf_list)):
  #   print(sort_trf_list[i],
  #           'len %i: %.5f +/- %.5f' % (len(sort_trf_list[i]), sort_means[i], sort_stds[i]))
