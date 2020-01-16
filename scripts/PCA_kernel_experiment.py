"""
JP PCA
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from parameters import general_keys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA as PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

CLASSES_NAMES = ['bogus', 'real']


def encircle(x, y, ax=None, **kw):
  if not ax: ax = plt.gca()
  p = np.c_[x, y]
  hull = ConvexHull(p)
  poly = plt.Polygon(p[hull.vertices, :], **kw)
  ax.add_patch(poly)


def scatter_2d(proyections, labels):
  # Scatterplot against PC1 and PC2
  fig, ax = plt.subplots(1, 1, figsize=(16, 12))
  labels_values = np.unique(labels)
  colors = ['blue', 'green']
  for label_i in labels_values:
    # Row masks for each category
    rows_0 = labels == label_i;

    # Plot
    lines = ax.scatter(
        proyections[rows_0, 0], proyections[rows_0, 1], edgecolor='k', s=120,
        label=CLASSES_NAMES[int(label_i)], alpha=0.5, c=colors[int(label_i)])
    # Encircle the boundaries
    encircle(proyections[rows_0, 0], proyections[rows_0, 1], fc="none",
             linewidth=5, ec=colors[int(label_i)])
    # Shading
    encircle(proyections[rows_0, 0], proyections[rows_0, 1],
             ec="k", alpha=0.05, fc=colors[int(label_i)])
  # Labels
  ax.set_title(
      "HiTS: Scatterplot of First Two PCA directions",
      fontsize=22)
  ax.set_xlabel("1st Principal Component", fontsize=22)
  ax.set_ylabel("2nd Principal Component", fontsize=22)
  ax.legend(loc='best', title='Transaction Type', fontsize=16)
  plt.show();


def plot_image(images, labels, n_images=1):
  for _ in range(n_images):
    index = np.random.randint(images.shape[0])
    n_channels = int(images.shape[-1])
    fig, ax = plt.subplots(1, n_channels, figsize=(3 * n_channels, 3))
    if type(ax) != np.ndarray:
      ax = [ax]
    for channel in range(n_channels):
      ax[channel].imshow(images[index, :, :, channel])
    plt.title(CLASSES_NAMES[int(labels[index])]+' '+str(labels[index]))
    plt.show()

def pca_experiment_inverse_error(n_data=200):
  """see inverse transform efect of proyection in original space"""
  pca = PCA(kernel='rbf', n_components=1)
  X_orig = np.random.multivariate_normal((0,0), ((10,2),(2,2)), n_data)#np.random.rand(n_data, 2)#np.stack([np.random.normal(0, 10, n_data), np.random.normal(0, 2, n_data)], axis=-1)
  X_re_orig = pca.inverse_transform(pca.fit_transform(X_orig))

  [plt.plot([X_re_orig[i, 0], X_re_orig[i+1, 0]], [X_re_orig[i, 1], X_re_orig[i+1, 1]], c='black')
   for i in range(n_data-1)]
  plt.scatter(X_orig[:, 0], X_orig[:, 1], label='Original points')
  plt.scatter(X_re_orig[:, 0], X_re_orig[:, 1], label='InverseTransform')
  [plt.plot([X_orig[i, 0], X_re_orig[i, 0]], [X_orig[i, 1], X_re_orig[i, 1]])
   for i in range(n_data)]
  plt.ylim((np.min([X_orig,X_re_orig])-1, np.max([X_orig,X_re_orig])+1))
  plt.xlim((np.min([X_orig,X_re_orig])-1, np.max([X_orig,X_re_orig])+1))
  plt.legend()
  plt.show()

  distances_test = np.sqrt(np.sum((X_re_orig - X_orig) ** 2, axis=-1))
  plt.hist(distances_test, bins=int(n_data/2), label='euclidean')
  plt.legend()
  plt.show()

  distances_0 = np.sqrt((X_re_orig[:,0] - X_orig[:,0]) ** 2)
  distances_1 = np.sqrt((X_re_orig[:,1] - X_orig[:,1]) ** 2)
  plt.hist(distances_0, bins=int(n_data/2), label='dim_0', alpha=0.5)
  plt.hist(distances_1, bins=int(n_data/2), label='dim_1', alpha=0.5)
  plt.legend()
  plt.show()


def pca_experiment_keep_dims():
  """see inverse transform efect of proyection in original space"""
  import numpy as np
  from sklearn.decomposition import PCA
  pca = PCA(kernel='rbf',n_components=2)
  X_orig = np.random.rand(10, 2)
  X_pca = pca.fit_transform(X_orig)

  plt.scatter(X_orig[:, 0], X_orig[:, 1], label='Original points')
  plt.scatter(X_pca[:, 0], X_pca[:, 1], label='PCA points')
  plt.legend()
  plt.show()

def get_every_dim_error_scores(n_pca_components, x_train_norm, x_test_norm, pca=None):
  if pca:
    pca = pca.fit_transform(x_train_norm)
  else:
    pca = PCA(kernel='rbf',n_components=n_pca_components).fit(x_train_norm)
  x_test_pca = pca.transform(x_test_norm)
  x_test_back = pca.inverse_transform(x_test_pca)
  score = np.zeros(x_test_norm.shape[0])
  for dim_i in range(x_test_back.shape[-1]):
    score += np.sqrt((x_test_back[:, dim_i] - x_test_norm[:, dim_i]) ** 2)
  return score

def get_every_dim_error_scores(n_pca_components, x_train_norm, x_test_norm, pca=None):
  if pca:
    pca = pca.fit_transform(x_train_norm)
  else:
    pca = PCA(kernel='rbf',n_components=n_pca_components).fit(x_train_norm)
  x_test_pca = pca.transform(x_test_norm)
  x_test_back = pca.inverse_transform(x_test_pca)
  score = np.zeros(x_test_norm.shape[0])
  for dim_i in range(x_test_back.shape[-1]):
    score += np.sqrt((x_test_back[:, dim_i] - x_test_norm[:, dim_i]) ** 2)
  return score

def get_proj_error_scores(n_pca_components, x_train_norm, x_test_norm, pca=None):
  if pca:
    pca = pca.fit_transform(x_train_norm)
  else:
    pca = PCA(kernel='rbf',n_components=n_pca_components).fit(x_train_norm)
  x_test_pca = pca.transform(x_test_norm)
  x_test_back = pca.inverse_transform(x_test_pca)
  score = np.sqrt(np.sum((x_test_back - x_test_norm)**2, axis=-1))
  return score

def get_every_proj_stat_test_scores(n_pca_components, x_train_norm, x_test_norm, percentil=97.73, pca=None):
  if pca:
    pca = pca.fit_transform(x_train_norm)
  else:
    pca = PCA(kernel='rbf',n_components=n_pca_components).fit(x_train_norm)
  x_train_pca = pca.transform(x_train_norm)
  thresholds = []
  print(x_train_pca.shape[-1])
  for dim_i in range(x_train_pca.shape[-1]):
    thr = np.percentile(x_train_pca[:, dim_i], percentil)
    thresholds.append(thr)
  x_test_pca = pca.transform(x_test_norm)
  score = np.sum(x_test_pca - thresholds, axis=-1)
  # score = np.zeros(x_test_norm.shape[0])
  # for dim_i in range(x_test_back.shape[-1]):
  #   score += np.sqrt((x_test_back[:, dim_i] - x_test_norm[:, dim_i]) ** 2)
  return score

def pca(X):
  # Data matrix X, assumes 0-centered
  n, m = X.shape
  assert np.allclose(X.mean(axis=0), np.zeros(m))
  # Compute covariance matrix
  C = np.dot(X.T, X) / (n-1)
  # Eigen decomposition
  eigen_vals, eigen_vecs = np.linalg.eig(C)
  # Project X onto PC space
  X_pca = np.dot(X, eigen_vecs)
  return X_pca

def svd(X):
  # Data matrix X, X doesn't need to be 0-centered
  n, m = X.shape
  # Compute full SVD
  U, Sigma, Vh = np.linalg.svd(X,
      full_matrices=False, # It's not necessary to compute the full matrix of U or V
      compute_uv=True)
  # Transform X with SVD components
  X_svd = np.dot(U, np.diag(Sigma))
  return X_svd

#https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
#https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

if __name__ == '__main__':
  from parameters import loader_keys
  from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
  import pandas as pd
  from scripts.detached_transformer_od_hits import \
    plot_histogram_disc_loss_acc_thr
  pca_experiment_inverse_error()
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
         PROJECT_PATH, '../datasets/outlier_seed42_crop21_nChannels4_HiTS2013_300k_samples.pkl'))#hits_outlier_dataset.get_outlier_detection_datasets()#
  print(x_train.shape)
  print(np.unique(y_test, return_counts=True))

  # Standardizing the features
  x_train_flat = x_train.reshape([x_train.shape[0], -1])
  print(x_train_flat.shape)
  print(x_train_flat[:, 0].mean(), x_train_flat[:, 0].std())
  scaler = StandardScaler().fit(x_train_flat)
  x_train_norm = scaler.transform(x_train_flat)
  print(x_train_norm[:, 0].mean(), x_train_norm[:, 0].std())
  pca = PCA(kernel='rbf',n_components=x_train_norm.shape[-1]).fit(x_train_norm)
  print("Varianza explicada por los primeros componentes principales:")
  print(pca.explained_variance_ratio_)
  print("Suma acumulada de los primeros componentes principales: %f" % np.sum(
      pca.explained_variance_ratio_))

  cum_sum_pca = np.cumsum(pca.explained_variance_ratio_)
  print(list(zip(np.arange(cum_sum_pca.shape[0]) + 1, cum_sum_pca)))
  thr_ind = np.argwhere(cum_sum_pca > 0.9)[0]
  print(list(
      zip((np.arange(cum_sum_pca.shape[0]) + 1)[thr_ind],
          cum_sum_pca[thr_ind])))

  x_test_flat = x_test.reshape([x_test.shape[0], -1])
  x_test_norm = scaler.transform(x_test_flat)

  # # Toy example
  # pca2 = PCA(kernel='rbf',n_components=2).fit(x_train_norm)
  # # x_train_pca2 = pca2.transform(x_train_flat)
  # x_test_pca2 = pca2.transform(x_test_flat)
  # scatter_2d(x_test_pca2, y_test)
  #
  # plot_image(x_test[y_test==0], y_test[y_test==0], n_images=3)
  # plot_image(x_test[y_test == 1], y_test[y_test == 1], n_images=3)


  # # Plotting errors This is incorrect, because inverse should be performed
  # x_train_pca = pca.transform(x_train_norm)
  # distances = np.sqrt(np.sum((x_train_norm - x_train_pca)**2, axis=-1))
  # plt.hist(distances, bins=100)
  # plt.show()
  # x_test_pca = pca.transform(x_test_norm)
  # distances_test = np.sqrt(np.sum((x_test_norm - x_test_pca)**2, axis=-1))
  # plt.hist(distances_test[y_test==0], bins=100, label=CLASSES_NAMES[0], alpha=0.5)
  # plt.hist(distances_test[y_test==1], bins=100, label=CLASSES_NAMES[1], alpha=0.5)
  # plt.legend()
  # plt.show()


  # # Plotting errors 90Var
  # pca_90 = PCA(kernel='rbf',n_components=thr_ind[0]).fit(x_train_norm)
  # print("Suma acumulada de los primeros componentes principales: %f" % np.sum(
  #     pca_90.explained_variance_ratio_))
  # x_train_pca_90 = pca_90.transform(x_train_norm)
  # x_train_back_90 = pca_90.inverse_transform(x_train_pca_90)
  # # print(x_train_back_90[0][:10]); print(x_train_norm[0][:10])
  # distances = np.sqrt(np.sum((x_train_back_90 - x_train_norm)**2, axis=-1))
  # plt.hist(distances, bins=100)
  # plt.show()
  # x_test_pca_90 = pca_90.transform(x_test_norm)
  # x_test_back_90 = pca_90.inverse_transform(x_test_pca_90)
  #
  # distances_test = np.sqrt(np.sum((x_test_back_90 - x_test_norm)**2, axis=-1))
  # plt.hist(distances_test[y_test==0], bins=100, label=CLASSES_NAMES[0], alpha=0.5)
  # plt.hist(distances_test[y_test==1], bins=100, label=CLASSES_NAMES[1], alpha=0.5)
  # plt.legend()
  # plt.show()
  #
  # pca_experiment_inverse_error()
  # pca_experiment_keep_dims()

  dim_error_score = get_every_dim_error_scores(thr_ind[0], x_train_norm, x_test_norm)
  plot_histogram_disc_loss_acc_thr(dim_error_score[y_test==1], dim_error_score[~(y_test==1)],
                                   x_label_name='DimError')

  proj_error_score = get_proj_error_scores(thr_ind[0], x_train_norm,
                                               x_test_norm)
  plot_histogram_disc_loss_acc_thr(proj_error_score[y_test == 1],
                                   proj_error_score[~(y_test == 1)],
                                   x_label_name='ProjError')

  stat_proj_score = get_every_proj_stat_test_scores(thr_ind[0], x_train_norm,
                                               x_test_norm)
  plot_histogram_disc_loss_acc_thr(stat_proj_score[y_test == 1],
                                   stat_proj_score[~(y_test == 1)],
                                   x_label_name='statProj')





