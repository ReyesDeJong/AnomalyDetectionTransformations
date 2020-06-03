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
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys
import numpy as np
from modules import utils

if __name__ == "__main__":
  MODEL_CHKP_PATH = os.path.join(PROJECT_PATH, 'results', 'Trf_Rank')
  RESULT_PATH = 'aux_results'
  utils.check_paths(RESULT_PATH)
  N_RUNS = 5
  N_TO_PLOT = 1
  SHOW_PLOT = True
  # SHOW_PLOT = False
  RANDOM_STATE = 42

  transformer = RankingTransformer(0, 0, 0, 0, 0, 0, 1, 0)
  print(transformer.transformation_tuples)
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
  data_loader_i = hits_loader

  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader_i.get_outlier_detection_datasets()
  inliers_test = x_test[y_test == 1]
  print(inliers_test.shape)
  outliers_test = x_test[y_test != 1]
  print(outliers_test.shape)

  # print('Visualize Data')
  # for i in range(N_TO_PLOT):
  #   idx = np.random.RandomState(RANDOM_STATE + i).randint(len(x_train))
  #   data_loader_i.plot_image(x_train[idx], show=SHOW_PLOT,
  #                            title='Train_%i' % idx)
  # for i in range(N_TO_PLOT):
  #   idx = np.random.RandomState(RANDOM_STATE + i).randint(len(x_val))
  #   data_loader_i.plot_image(x_val[idx], show=SHOW_PLOT, title='Val_%i' % idx)
  # for i in range(N_TO_PLOT):
  #   idx = np.random.RandomState(RANDOM_STATE + i).randint(len(x_test))
  #   data_loader_i.plot_image(x_test[idx], show=SHOW_PLOT, title='Test_%i' % idx)

  # for i in range(N_TO_PLOT):
  #   idx = np.random.RandomState(RANDOM_STATE + i).randint(len(inliers_test))
  #   data_loader_i.plot_image(inliers_test[idx], show=SHOW_PLOT,
  #                            title='Inliers_Test_%i' % idx)
  # for i in range(N_TO_PLOT):
  #   idx = np.random.RandomState(RANDOM_STATE + i).randint(len(outliers_test))
  #   data_loader_i.plot_image(outliers_test[idx], show=SHOW_PLOT,
  #                            title='Outliers_Test_%i' % idx)

  print('\n Transform Visualizer')
  data_trf_type = {'inlier': inliers_test,
                   'outlier': outliers_test,
                   }
  # for data_type in data_trf_type.keys():
  #   for trf_value_i in range(transformer.n_transforms):
  #     trf_i_data, _ = transformer.apply_transforms(data_trf_type[data_type],
  #                                                  [trf_value_i])
  #     for i in range(N_TO_PLOT):
  #       idx = np.random.RandomState(RANDOM_STATE + i).randint(
  #           len(trf_i_data))
  #       data_loader_i.plot_image(
  #           trf_i_data[idx], show=SHOW_PLOT, title='%s_Test_trf_%s_%i' % (
  #             data_type, str(transformer.transformation_tuples[trf_value_i]),
  #             idx))

  data_1126 = outliers_test[1126]
  data_loader_i.plot_image(
      data_1126, show=SHOW_PLOT)
  trf_data_1126 = transformer.apply_transforms(data_1126[None,...], [1])[0][0]
  data_loader_i.plot_image(
      trf_data_1126, show=SHOW_PLOT)

  data_1126 = outliers_test[1126:1129]
  data_loader_i.plot_image(
      data_1126[0], show=SHOW_PLOT)
  trf_data_1126 = transformer.apply_transforms(data_1126, [1])[0]
  data_loader_i.plot_image(
      trf_data_1126[0], show=SHOW_PLOT)

  trf_outliers, trf_outlier_lbl = transformer.apply_all_transforms(outliers_test)

  for i in range(N_TO_PLOT):
    idx = np.random.RandomState(RANDOM_STATE + i).randint(
                  len(outliers_test))
    data_loader_i.plot_image(
        outliers_test[idx], show=SHOW_PLOT, title='outliers_Test_%i' % (
          idx))

  trf_0_idxs = np.where(trf_outlier_lbl == 0)[0]
  tfr_0_outliers = trf_outliers[trf_outlier_lbl == 0]
  # tfr_0_outliers,_ = transformer.apply_transforms(outliers_test, [0])
  for i in range(N_TO_PLOT):
    idx = np.random.RandomState(RANDOM_STATE + i).randint(
        len(tfr_0_outliers))
    data_loader_i.plot_image(
        tfr_0_outliers[idx], show=SHOW_PLOT, title='outliers_trf0_Test_%i' % (
          idx))

  trf_2_idxs = np.where(trf_outlier_lbl == 1)[0]
  tfr_2_outliers = trf_outliers[trf_outlier_lbl == 1]
  # tfr_2_outliers,_ = transformer.apply_transforms(outliers_test, [1])
  for i in range(N_TO_PLOT):
    idx = np.random.RandomState(RANDOM_STATE + i).randint(
        len(tfr_2_outliers))
    data_loader_i.plot_image(
        tfr_2_outliers[idx], show=SHOW_PLOT, title='outliers_trf2_Test_%i' % (
          idx))

  # for loader_i in data_loaders:
  #   (x_train, y_train), (x_val, y_val), (
  #     x_test, y_test) = loader_i.get_outlier_detection_datasets()
  #   gt_outliers = x_test[y_test != 1]
  #   mnist = prepare_images(get_dataset('mnist', int(len(x_test) // 2)), x_train)
  #   # print(mnist.shape)
  #   # print(np.mean(np.max(mnist, axis=(1, 2)), axis=0))
  #   # print(np.mean(np.min(mnist, axis=(1, 2)), axis=0))
  #   cifar10 = prepare_images(get_dataset('cifar10', int(len(x_test) // 2)),
  #                            x_train)
  #   shuffle_trf = RankingTransformer(0, 0, 0, 0, 0, 0, 0)
  #   inliers = x_test[y_test == 1]
  #   inliers_shuffle, _ = shuffle_trf.apply_transforms(inliers, [1])
  #   # print(inliers_shuffle.shape)
  #   if loader_i.name == hits_loader.name:
  #     _, _, (
  #       x_test_other,
  #       y_test_other) = ztf_loader.get_outlier_detection_datasets()
  #     other_set_outliers = x_test_other[y_test_other != 1][
  #                          :int(len(x_test) // 2)]
  #   else:
  #     hits_params = {
  #       loader_keys.DATA_PATH: os.path.join(
  #           PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
  #       loader_keys.N_SAMPLES_BY_CLASS: 10000,
  #       loader_keys.TEST_PERCENTAGE: 0.3,
  #       loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
  #       loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
  #       loader_keys.CROP_SIZE: 21,
  #       general_keys.RANDOM_SEED: 42,
  #       loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  #     }
  #     hits_loader = HiTSOutlierLoader(hits_params, pickles_usage=False)
  #     _, _, (
  #       x_test_other,
  #       y_test_other) = hits_loader.get_outlier_detection_datasets()
  #     other_set_outliers = x_test_other[y_test_other != 1][
  #                          :int(len(x_test) // 2)]
  #   other_set_outliers = prepare_images(other_set_outliers, x_train)
  #   # print(other_set_outliers.shape)
  #
  #   outliers_dict = {
  #     'gt': gt_outliers,
  #     'shuffle': inliers_shuffle,
  #     'other_astro': other_set_outliers,
  #     'mnist': mnist,
  #     'cifar10': cifar10
  #   }
  #
  #   pickle_name = 'small_rank_%s.pkl' % loader_i.name
  #   pickle_name = os.path.join(RESULT_PATH, pickle_name)
  #   result_all_runs = {}
  #   for run_i in range(N_RUNS):
  #     indexes_for_power_set = list(
  #         range(len(power_set_clean)))  # + [-1, -2, -3,
  #     #    -4]
  #     # indexes_for_power_set = list(range(len(power_set_clean)))[:3] + [-1]
  #     result_each_trf = {}
  #     for power_set_idx in indexes_for_power_set:
  #       if power_set_idx == -1:
  #         model = TransformODModel(
  #             loader_i, trf_72, input_shape=x_train.shape[1:],
  #             results_folder_name=MODEL_CHKP_PATH)
  #         trf_to_perform = np.array(trf_72.transformation_tuples)
  #       elif power_set_idx == -2:
  #         model = TransformODModel(
  #             loader_i, trf_9, input_shape=x_train.shape[1:],
  #             results_folder_name=MODEL_CHKP_PATH)
  #         trf_to_perform = np.array(trf_9.transformation_tuples)
  #       elif power_set_idx == -3:
  #         model = TransformODSimpleModel(
  #             loader_i, trf_72, input_shape=x_train.shape[1:],
  #             results_folder_name=MODEL_CHKP_PATH)
  #         trf_to_perform = np.array(trf_72.transformation_tuples)
  #       elif power_set_idx == -4:
  #         model = TransformODSimpleModel(
  #             loader_i, trf_9, input_shape=x_train.shape[1:],
  #             results_folder_name=MODEL_CHKP_PATH)
  #         trf_to_perform = np.array(trf_9.transformation_tuples)
  #       else:
  #         trforms_indx_set = power_set_clean[power_set_idx]
  #         trf_to_perform = np.array(aux_transformer.transformation_tuples)[
  #           np.array(trforms_indx_set)]
  #         print(trf_to_perform)
  #         trfer = RankingTransformer()
  #         trfer.set_transformations_to_perform(trf_to_perform.tolist())
  #         model = TransformODSimpleModel(
  #             loader_i, trfer, input_shape=x_train.shape[1:],
  #             results_folder_name=MODEL_CHKP_PATH)
  #       model.fit(x_train, x_val, epochs=1000, patience=0)
  #       result_each_outliers = {}
  #       for outlier_key in outliers_dict.keys():
  #         current_outliers = outliers_dict[outlier_key]
  #         # print(inliers.shape)
  #         # print(current_outliers.shape)
  #         current_x_test = np.concatenate([inliers, current_outliers], axis=0)
  #         current_y_test = np.concatenate(
  #             [y_test[y_test == 1], y_test[y_test != 1]], axis=0)
  #         results = model.evaluate_od(
  #             x_train, current_x_test, current_y_test,
  #             '%s_%s' % (loader_i.name, outlier_key), 'real', x_val)
  #         print('%i %s_%s %.5f' % (
  #           len(trf_to_perform), loader_i.name, outlier_key,
  #           results['dirichlet']['roc_auc']))
  #         result_each_outliers[outlier_key] = [trf_to_perform, results]
  #       result_each_trf[power_set_idx] = result_each_outliers
  #     result_all_runs[run_i] = result_each_trf
  #     save_pickle(result_all_runs, pickle_name)
  #   save_pickle(result_all_runs, pickle_name)
