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
from sklearn.decomposition import PCA
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

def pca_experiment():
  """see inverse transform efect of proyection in original space"""
  import numpy as np
  from sklearn.decomposition import PCA
  pca = PCA(1)
  X_orig = np.random.rand(10, 2)
  X_re_orig = pca.inverse_transform(pca.fit_transform(X_orig))

  plt.scatter(X_orig[:, 0], X_orig[:, 1], label='Original points')
  plt.scatter(X_re_orig[:, 0], X_re_orig[:, 1], label='InverseTransform')
  [plt.plot([X_orig[i, 0], X_re_orig[i, 0]], [X_orig[i, 1], X_re_orig[i, 1]])
   for i in range(10)]
  plt.legend()
  plt.show()


if __name__ == '__main__':
  from parameters import loader_keys
  from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader

  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2, 3],#[2],  #
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  hits_outlier_dataset = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = hits_outlier_dataset.get_outlier_detection_datasets()
  print(x_train.shape)
  print(np.unique(y_test, return_counts=True))

  # Standardizing the features
  x_train_flat = x_train.reshape([x_train.shape[0], -1])
  print(x_train_flat.shape)
  print(x_train_flat[:, 0].mean(), x_train_flat[:, 0].std())
  scaler = StandardScaler().fit(x_train_flat)
  x_train_norm = scaler.transform(x_train_flat)
  print(x_train_norm[:, 0].mean(), x_train_norm[:, 0].std())
  pca = PCA(n_components=x_train_norm.shape[-1]).fit(x_train_norm)
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
  # pca2 = PCA(n_components=2).fit(x_train_norm)
  # # x_train_pca2 = pca2.transform(x_train_flat)
  # x_test_pca2 = pca2.transform(x_test_flat)
  # scatter_2d(x_test_pca2, y_test)
  #
  # plot_image(x_test[y_test==0], y_test[y_test==0], n_images=3)
  # plot_image(x_test[y_test == 1], y_test[y_test == 1], n_images=3)


  # Plotting errors
  x_train_pca = pca.transform(x_train_norm)
  distances = np.sqrt(np.sum((x_train_norm - x_train_pca)**2, axis=-1))
  plt.hist(distances, bins=100)
  plt.show()
  x_test_pca = pca.transform(x_test_norm)
  distances_test = np.sqrt(np.sum((x_test_norm - x_test_pca)**2, axis=-1))
  plt.hist(distances_test[y_test==0], bins=100, label=CLASSES_NAMES[0], alpha=0.5)
  plt.hist(distances_test[y_test==1], bins=100, label=CLASSES_NAMES[1], alpha=0.5)
  plt.legend()
  plt.show()


  # # Plotting errors 90Var
  # pca_90 = PCA(n_components=thr_ind[0]+1).fit(x_train_norm)
  # print("Suma acumulada de los primeros componentes principales: %f" % np.sum(
  #     pca_90.explained_variance_ratio_))
  # x_train_pca_90 = pca_90.transform(x_train_norm)
  # x_train_back_90 = pca_90.inverse_transform(x_train_pca_90)
  # print(x_train_back_90[0][:10]); print(x_train_norm[0][:10])
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
  # pca_experiment()





