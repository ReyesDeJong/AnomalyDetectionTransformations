"""
JP PCA
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

CLASSES_NAMES = ['bogus', 'real']

def get_every_proj_stat_test_scores(n_pca_components, x_train_norm, x_test_norm,
    percentil=97.73, pca=None, x_val_norm=None, return_train=False):
  if pca:
    pca = pca.fit_transform(x_train_norm)
  else:
    pca = PCA(n_components=n_pca_components).fit(x_train_norm)
  x_train_pca = pca.transform(x_train_norm)
  thresholds = []
  print(x_train_pca.shape[-1])
  print("Suma acumulada de los primeros componentes principales: %f" % np.sum(
      pca.explained_variance_ratio_))
  for dim_i in range(x_train_pca.shape[-1]):
    thr = np.percentile(x_train_pca[:, dim_i], percentil)
    thresholds.append(thr)
  x_test_pca = pca.transform(x_test_norm)
  scores = x_test_pca - thresholds
  sum_score = np.sum(scores, axis=-1)
  # score = np.zeros(x_test_norm.shape[0])
  # for dim_i in range(x_test_back.shape[-1]):
  #   score += np.sqrt((x_test_back[:, dim_i] - x_test_norm[:, dim_i]) ** 2)
  if x_val_norm is not None:
    x_val_pca = pca.transform(x_val_norm)
    scores_val = x_val_pca - thresholds
    if return_train:
      scores_train = x_train_pca - thresholds
      return scores, scores_val, scores_train
    return scores, scores_val
  if return_train:
    scores_train = x_train_pca - thresholds
    return scores, scores_train
  return sum_score, scores


# https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
# https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

def _train_ocsvm_and_score(params, x_train, val_labels, x_val):
  return np.mean(val_labels ==
                 OneClassSVM(**params).fit(x_train).predict(
                     x_val))


if __name__ == '__main__':
  import pandas as pd
  from scripts.detached_transformer_od_hits import \
    plot_histogram_disc_loss_acc_thr

  # pca_experiment_inverse_error()
  # hits_params = {
  #   loader_keys.DATA_PATH: os.path.join(
  #       PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
  #   loader_keys.N_SAMPLES_BY_CLASS: 10000,
  #   loader_keys.TEST_PERCENTAGE: 0.2,
  #   loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
  #   loader_keys.USED_CHANNELS: [0, 1, 2, 3],#[2],  #
  #   loader_keys.CROP_SIZE: 21,
  #   general_keys.RANDOM_SEED: 42,
  #   loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  # }
  # hits_outlier_dataset = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = pd.read_pickle(os.path.join(
      PROJECT_PATH,
      '../datasets/outlier_seed42_crop21_nChannels4_HiTS2013_300k_samples.pkl'))  # hits_outlier_dataset.get_outlier_detection_datasets()#
  print(x_train.shape)
  print(np.unique(y_test, return_counts=True))

  # Standardizing the features
  x_train_flat = x_train.reshape([x_train.shape[0], -1])
  print(x_train_flat.shape)
  print(x_train_flat[:, 0].mean(), x_train_flat[:, 0].std())
  scaler = StandardScaler().fit(x_train_flat)
  x_train_norm = scaler.transform(x_train_flat)
  print(x_train_norm[:, 0].mean(), x_train_norm[:, 0].std())
  x_test_flat = x_test.reshape([x_test.shape[0], -1])
  x_test_norm = scaler.transform(x_test_flat)

  from sklearn.svm import OneClassSVM

  x_val_flat = x_val.reshape([x_val.shape[0], -1])
  x_val_norm = scaler.transform(x_val_flat)
  stat_proj_score_test, stat_proj_score_val, stat_proj_score_train = get_every_proj_stat_test_scores(
      0.9, x_train_norm, x_test_norm, return_train=True, x_val_norm=x_val_norm)
  best_params = {'gamma': 0.0078125, 'nu': 0.4}
  best_ocsvm = OneClassSVM(**best_params).fit(stat_proj_score_train)
  od_scores_test = best_ocsvm.decision_function(stat_proj_score_test)
  plot_histogram_disc_loss_acc_thr(
      od_scores_test[y_test == 1], od_scores_test[~(y_test == 1)],
      x_label_name='statProjAllDim-SVM')
