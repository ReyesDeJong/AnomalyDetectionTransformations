# %%

import os
import sys

import numpy as np

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname('__file__'), '..', '..'))
sys.path.append(PROJECT_PATH)

from parameters import general_keys
from parameters import loader_keys
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.geometric_transform.transformer_no_compositions import \
  NoCompositionTransformer
from modules.geometric_transform.transformations_tf import Transformer
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader

if __name__ == "__main__":
  SHOW_PLOT = True
  # RANDOM_SEED = np.random.randint(1e4)#
  RANDOM_SEED = 89
  # load data
  params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2, 3],
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  hits_outlier_loader = HiTSOutlierLoader(params)

  ztf_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH,
        '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
  }
  ztf_loader = ZTFSmallOutlierLoader(ztf_params)

  # data_loader = hits_outlier_loader
  data_loader = ztf_loader
  (X_train, y_train), (X_val, y_val), (
    X_test, y_test) = data_loader.get_outlier_detection_datasets()

  # print('Plot bogus-real')
  # n_inliers_int_test = int(len(y_test) / 2)
  # inlier_idx = np.random.RandomState(RANDOM_SEED).randint(n_inliers_int_test)
  # inlier_image = X_test[inlier_idx]
  # outlier_idx = np.random.RandomState(RANDOM_SEED).randint(n_inliers_int_test,
  #                                                          len(y_test))
  # outlier_image = X_test[outlier_idx]
  #
  # inliers_title = 'Inlier %i, minC_mean: %.2f, maxC_mean: %.2f' % (
  #   inlier_idx, np.mean(np.min(inlier_image, axis=(0, 1))),
  #   np.mean(np.max(inlier_image, axis=(0, 1))))
  # print('Inlier %i, maxC_mean: %s, minC_mean: %s' % (
  #   inlier_idx, str(np.max(inlier_image, axis=(0, 1))),
  #   str(np.min(inlier_image, axis=(0, 1)))))
  # outliers_title = 'outlier %i, minC_mean: %.2f, maxC_mean: %.2f' % (
  #   outlier_idx, np.mean(np.min(outlier_image, axis=(0, 1))),
  #   np.mean(np.max(outlier_image, axis=(0, 1))))
  # print('outlier %i, maxC_mean: %s, minC_mean: %s' % (
  #   outlier_idx, str(np.max(outlier_image, axis=(0, 1))),
  #   str(np.min(outlier_image, axis=(0, 1)))))
  # data_loader.plot_image(inlier_image, show=SHOW_PLOT,
  #                        title=inliers_title)
  # data_loader.plot_image(outlier_image, show=SHOW_PLOT,
  #                        title=outliers_title)

  # transformer = NoCompositionTransformer()
  x_data = X_test
  transformer = Transformer()
  x_transforms, trf_idexes = transformer.apply_all_transforms(x_data)
  sample_idx = np.random.RandomState(RANDOM_SEED).randint(len(x_data))
  data_loader.plot_image(x_data[sample_idx], show=SHOW_PLOT, title='original')
  trf_idexes_to_plot = np.arange(transformer.n_transforms)
  np.random.RandomState(RANDOM_SEED).shuffle(trf_idexes_to_plot)
  trf_idexes_to_plot = trf_idexes_to_plot[:10]
  for i in trf_idexes_to_plot:
    trf_tuple = transformer.transformation_tuples[i]
    inlier_image_trf = x_transforms[trf_idexes==i][sample_idx]


    print('\nTransformation %i: %s' % (i, transformer.transformation_tuples[i]))
    inliers_title = 'Inlier trf%i:%s, minC_mean: %.2f, maxC_mean: %.2f' % (
      i, trf_tuple, np.mean(np.min(inlier_image_trf, axis=(0, 1))),
      np.mean(np.max(inlier_image_trf, axis=(0, 1))))
    data_loader.plot_image(inlier_image_trf, show=SHOW_PLOT,
                           title=inliers_title)

  # for i in trf_idexes_to_plot:
  #   trf_tuple = transformer.transformation_tuples[i]
  #   x_trf_i = x_transforms[trf_idexes == i]
  #   x_trf_applied ,_ = transformer.apply_transforms(x_data,
  #                                                   transformations_inds=[i])
  #   print('Trf %i %s: %f' % (i, trf_tuple, np.mean(x_trf_i==x_trf_applied)))