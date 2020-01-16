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
from sklearn import metrics
from sklearn.svm import OneClassSVM

CLASSES_NAMES = ['bogus', 'real']

# https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
# https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8
def plot_image(images, labels, n_images=1):
  for _ in range(n_images):
    index = np.random.randint(images.shape[0])
    n_channels = int(images.shape[-1])
    fig, ax = plt.subplots(1, n_channels, figsize=(3 * n_channels, 3))
    if type(ax) != np.ndarray:
      ax = [ax]
    for channel in range(n_channels):
      ax[channel].imshow(images[index, :, :, channel])
    plt.title(CLASSES_NAMES[int(labels[index])] + ' ' + str(labels[index]))
    plt.show()


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
      '../datasets/generated_hits_4_channels/outlier_seed42_crop21_nChannels4_HiTS2013_300k_samples.pkl'))  # hits_outlier_dataset.get_outlier_detection_datasets()#

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
  # best_ocsvm = OneClassSVM(**best_params).fit(np.concatenate([scores_train, scores_val]))
  best_ocsvm = OneClassSVM(**best_params).fit(scores_train)
  od_scores_test = best_ocsvm.decision_function(scores_test)
  od_scores_train = best_ocsvm.decision_function(scores_train)
  od_scores_val = best_ocsvm.decision_function(scores_val)

  plt.hist(od_scores_train, bins=100, label='train', alpha=0.5, density=True)
  plt.hist(od_scores_val, bins=100, label='val', alpha=0.5, density=True)
  plt.legend()
  plt.show()
  plt.hist(od_scores_test[y_test==0], bins=100, label='test_'+CLASSES_NAMES[0], alpha=0.5, density=True)
  plt.hist(od_scores_test[y_test == 1], bins=100,
           label='test_' + CLASSES_NAMES[1], alpha=0.5, density=True)
  plt.legend()
  plt.show()
  plot_histogram_disc_loss_acc_thr(
      od_scores_test[y_test == 1], od_scores_test[~(y_test == 1)],
      x_label_name='statProjAllDim-SVM')

  X_test = np.concatenate([scores_test[y_test == 1], scores_test[~(y_test == 1)]], axis=0)

  pred = best_ocsvm.decision_function(X_test)
  y_true = np.array([1] * np.sum(y_test == 1) + [-1] * np.sum(~(y_test == 1)))
  fpr, tpr, thresholds = metrics.roc_curve(y_true, pred)
  roc_auc = metrics.auc(fpr, tpr)

  plt.figure()
  plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.show()


  pca2 = PCA(2).fit(scores_train)
  x_train_pca2 = pca2.transform(scores_train)
  x_test_pca2 = pca2.transform(scores_test)
  x_val_pca2 = pca2.transform(scores_val)

  plt.scatter(x_train_pca2[:,0], x_train_pca2[:, 1], label='train', alpha=0.5)
  plt.scatter(x_test_pca2[y_test == 1, 0], x_test_pca2[y_test == 1, 1], label='test_in', alpha=0.5)
  plt.scatter(x_val_pca2[:, 0], x_val_pca2[:, 1], label='val', alpha=0.5)
  plt.legend()
  plt.show()

  # plt.scatter(x_train_pca2[:,0], x_train_pca2[:,1], label='train', alpha=0.5)
  plt.scatter(x_test_pca2[y_test == 1, 0], x_test_pca2[y_test == 1, 1], label='test_in', alpha=0.5)
  plt.scatter(x_test_pca2[y_test == 0, 0], x_test_pca2[y_test == 0, 1],
              label='test_out', alpha=0.5)
  # plt.scatter(x_val_pca2[:, 0], x_val_pca2[:, 1], label='val', alpha=0.5)
  plt.legend()
  plt.show()



  # matches = 0
  # for train_i in range(x_val_flat.shape[0]):
  #   train_sample = x_val_flat[train_i]
  #   for test_i in range(x_test_flat.shape[0]):
  #     test_sample = x_test_flat[test_i]
  #     if np.mean(test_sample==train_sample)==1:
  #       matches+=1
  #       print(matches)
  # train_sample = x_train_flat[0]
  # for test_i in range(x_test_flat.shape[0]):
  #   test_sample = x_test_flat[test_i]
  #   if np.mean(test_sample==train_sample)==1:
  #     print(test_i)

  plot_image(x_train[0][None,...], labels=[1])
  plot_image(x_test[0][None, ...], labels=[1])