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
from modules import utils
import tensorflow_datasets as tfds
import tensorflow as tf


def get_powerset(iterable):
  "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  s = list(iterable)
  return list(
      chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def prepare_images(unprepraed_images, images_example):
  if unprepraed_images.shape[1] != images_example.shape[1] or \
      unprepraed_images.shape[2] != images_example.shape[2]:
    unprepraed_images = tf.image.resize(unprepraed_images,
                                        images_example.shape[1:3]).numpy()
  if unprepraed_images.shape[-1] <= images_example.shape[-1]:
    while unprepraed_images.shape[-1] != images_example.shape[-1]:
      unprepraed_images = np.concatenate(
          [unprepraed_images, unprepraed_images[..., -1][..., None]], axis=-1)
  else:
    unprepraed_images = unprepraed_images[..., :images_example.shape[-1]]
  unprepraed_images = utils.normalize_by_channel_1_1(unprepraed_images)
  return unprepraed_images


def main():
  MODEL_CHKP_PATH = os.path.join(PROJECT_PATH, 'results', 'Trf_Rank_Incremental')
  RESULT_PATH = 'aux_results'
  utils.check_paths(RESULT_PATH)
  METRIC_TO_RANK_ON = 'roc_auc'
  N_RUNS = 1

  aux_transformer = RankingTransformer()
  trf_72 = Transformer()
  trf_9 = TransTransformer()
  n_tuple_array = list(range(aux_transformer.n_transforms))
  power_set = get_powerset(n_tuple_array)
  power_set_clean = [x for x in power_set if
                     len(x) > 1 and 0 in x and len(x) < 4]
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

  data_loaders = [#hits_loader,
                  ztf_loader]

  for loader_i in data_loaders:
    (x_train, y_train), (x_val, y_val), (
      x_test, y_test) = loader_i.get_outlier_detection_datasets()
    gt_outliers = x_test[y_test != 1]
    # mnist = prepare_images(get_dataset('mnist', int(len(x_test) // 2)), x_train)
    # print(mnist.shape)
    # print(np.mean(np.max(mnist, axis=(1, 2)), axis=0))
    # print(np.mean(np.min(mnist, axis=(1, 2)), axis=0))
    # cifar10 = prepare_images(get_dataset('cifar10', int(len(x_test) // 2)),
    #                          x_train)
    # shuffle_trf = RankingTransformer(0, 0, 0, 0, 0, 0, 0)
    inliers = x_test[y_test == 1]
    # inliers_shuffle, _ = shuffle_trf.apply_transforms(inliers, [1])
    # print(inliers_shuffle.shape)
    if loader_i.name == hits_loader.name:
      _, _, (
        x_test_other,
        y_test_other) = ztf_loader.get_outlier_detection_datasets()
      other_set_outliers = x_test_other[y_test_other != 1][
                           :int(len(x_test) // 2)]
    else:
      hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 10000,
        loader_keys.TEST_PERCENTAGE: 0.3,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
      }
      hits_loader = HiTSOutlierLoader(hits_params, pickles_usage=False)
      _, _, (
        x_test_other,
        y_test_other) = hits_loader.get_outlier_detection_datasets()
      other_set_outliers = x_test_other[y_test_other != 1][
                           :int(len(x_test) // 2)]
    other_set_outliers = prepare_images(other_set_outliers, x_train)
    # print(other_set_outliers.shape)

    outliers_dict = {
      # 'gt': gt_outliers,
      # 'shuffle': inliers_shuffle,
      'other_astro': other_set_outliers,
      # 'mnist': mnist,
      # 'cifar10': cifar10
    }

    pickle_name = 'rank_incremental_%s_%s.pkl' % (loader_i.name, list(outliers_dict.keys())[0])
    pickle_name = os.path.join(RESULT_PATH, pickle_name)
    trfer = RankingTransformer()
    trf_list = trfer.transformation_tuples
    trf_t0 = [trf_list[0]]
    trf_to_rank = trf_list[1:]
    # print(trf_t0)
    # print(trf_to_rank)
    result_all_runs = {}
    for run_i in range(N_RUNS):
      results_run_i = {}
      best_rank_metric_so_far = 0
      for n_transforms_selected_so_far in range(len(trf_to_rank)):
        best_rank_found = 0
        for trf_i_idx in range(len(trf_to_rank)):
          trf_to_perform = trf_t0 + [trf_to_rank[trf_i_idx]]
          print(trf_to_perform)
          trfer.set_transformations_to_perform(trf_to_perform)
          model = TransformODSimpleModel(
              loader_i, trfer, input_shape=x_train.shape[1:],
              results_folder_name=MODEL_CHKP_PATH)
          model.fit(x_train, x_val, epochs=1000, patience=0)
          result_each_outliers = {}
          outlier_key = list(outliers_dict.keys())[0]
          current_outliers = outliers_dict[outlier_key]
          # print(inliers.shape)
          # print(current_outliers.shape)
          current_x_test = np.concatenate([inliers, current_outliers], axis=0)
          current_y_test = np.concatenate(
              [y_test[y_test == 1], y_test[y_test != 1]], axis=0)
          results = model.evaluate_od(
              x_train, current_x_test, current_y_test,
              '%s_%s' % (loader_i.name, outlier_key), 'real', x_val)
          print('%i %s_%s %.5f' % (
            len(trf_to_perform), loader_i.name, outlier_key,
            results['dirichlet']['roc_auc']))
          model_result_metric = results['dirichlet']['roc_auc']
          if model_result_metric > best_rank_metric_so_far:
            best_rank_metric_so_far = model_result_metric
            best_rank_found = 1
            best_trf = trf_to_perform
            print('best ', best_trf)
          results_run_i[trf_i_idx] = [trf_to_perform, results]
        if best_rank_found == 0:
          print('Best Trf %s: %f' % (str(best_trf), best_rank_metric_so_far))
          if 'gtxcvx' not in outlier_key:
            trfer.set_transformations_to_perform(best_trf)
            model = TransformODSimpleModel(
                loader_i, trfer, input_shape=x_train.shape[1:],
                results_folder_name=MODEL_CHKP_PATH)
            model.fit(x_train, x_val, epochs=1000, patience=0)
            current_x_test = np.concatenate([inliers, gt_outliers], axis=0)
            current_y_test = np.concatenate(
                [y_test[y_test == 1], y_test[y_test != 1]], axis=0)
            results = model.evaluate_od(
                x_train, current_x_test, current_y_test,
                '%s_%s' % (loader_i.name, outlier_key), 'real', x_val)
            print('%i %s_%s %.5f' % (
              len(best_trf), loader_i.name, 'gt_outliers',
              results['dirichlet']['roc_auc']))
          break
        trf_t0 = best_trf
        print(trf_t0)
        trf_to_rank = [x for x in trf_to_rank if x not in trf_t0]
        print(trf_to_rank)
      result_all_runs[run_i] = results_run_i
    #   save_pickle(result_all_runs, pickle_name)
    # save_pickle(result_all_runs, pickle_name)



if __name__ == "__main__":
  print('')
  main()
