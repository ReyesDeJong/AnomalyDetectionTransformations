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
from sklearn.svm import OneClassSVM

CLASSES_NAMES = ['bogus', 'real']

# https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
# https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

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
  x_val_flat = x_val.reshape([x_val.shape[0], -1])
  x_val_norm = scaler.transform(x_val_flat)

  pca = PCA(0.9).fit(x_train_norm)
  x_train_pca = pca.transform(x_train_norm)
  print(x_train_pca.shape)
  x_test_pca = pca.transform(x_test_norm)
  print(x_test_pca.shape)
  x_val_pca = pca.transform(x_val_norm)
  print(x_val_pca.shape)

  # critical values
  critical_value = 97.73
  n_components = x_train_pca.shape[-1]
  thresholds = []
  for component_i in range(n_components):
    thr = np.percentile(x_train_pca[:, component_i], critical_value)
    thresholds.append(thr)

  scores_train = x_train_pca - thresholds
  scores_test = x_test_pca - thresholds
  scores_val = x_val_pca - thresholds

  best_params = {'gamma': 0.0078125, 'nu': 0.4}
  best_ocsvm = OneClassSVM(**best_params).fit(stat_proj_score_train)
  od_scores_test = best_ocsvm.decision_function(stat_proj_score_test)
  plot_histogram_disc_loss_acc_thr(
      od_scores_test[y_test == 1], od_scores_test[~(y_test == 1)],
      x_label_name='statProjAllDim-SVM')
