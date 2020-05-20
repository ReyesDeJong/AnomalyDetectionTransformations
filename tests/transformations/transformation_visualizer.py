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
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader

if __name__ == "__main__":
  SHOW_PLOT = True
  # RANDOM_SEED = np.random.randint(1e4)#
  RANDOM_SEED = 45
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

  print('Plot bogus-real')
  n_inliers_int_test = int(len(y_test) / 2)
  inlier_idx = np.random.RandomState(RANDOM_SEED).randint(n_inliers_int_test)
  inlier_image = X_test[inlier_idx]
  outlier_idx = np.random.RandomState(RANDOM_SEED).randint(n_inliers_int_test,
                                                           len(y_test))
  outlier_image = X_test[outlier_idx]

  inliers_title = 'Inlier %i, minC_mean: %.2f, maxC_mean: %.2f' % (
    inlier_idx, np.mean(np.min(inlier_image, axis=(0, 1))),
    np.mean(np.max(inlier_image, axis=(0, 1))))
  print('Inlier %i, maxC_mean: %s, minC_mean: %s' % (
    inlier_idx, str(np.max(inlier_image, axis=(0, 1))),
    str(np.min(inlier_image, axis=(0, 1)))))
  outliers_title = 'outlier %i, minC_mean: %.2f, maxC_mean: %.2f' % (
    outlier_idx, np.mean(np.min(outlier_image, axis=(0, 1))),
    np.mean(np.max(outlier_image, axis=(0, 1))))
  print('outlier %i, maxC_mean: %s, minC_mean: %s' % (
    outlier_idx, str(np.max(outlier_image, axis=(0, 1))),
    str(np.min(outlier_image, axis=(0, 1)))))
  data_loader.plot_image(inlier_image, show=SHOW_PLOT,
                         title=inliers_title)
  data_loader.plot_image(outlier_image, show=SHOW_PLOT,
                         title=outliers_title)

  transformer = NoCompositionTransformer()
  for i in range(transformer.n_transforms):
    trf_tuple = transformer.transformation_tuples[i]
    inlier_image_trf = transformer.apply_transforms(inlier_image[None, ...],
                                                    transformations_inds=[i])[0][0]
    outlier_image_trf = transformer.apply_transforms(outlier_image[None, ...],
                                                     transformations_inds=[i])[
      0][0]

    print('\nTransformation %i: %s' % (i, transformer.transformation_tuples[i]))
    print(inlier_image_trf.shape)
    inliers_title = 'Inlier trf:%s, minC_mean: %.2f, maxC_mean: %.2f' % (
      trf_tuple, np.mean(np.min(inlier_image_trf, axis=(0, 1))),
      np.mean(np.max(inlier_image_trf, axis=(0, 1))))
    print('Inlier %i, maxC_mean: %s, minC_mean: %s' % (
      inlier_idx, str(np.max(inlier_image_trf, axis=(0, 1))),
      str(np.min(inlier_image_trf, axis=(0, 1)))))
    outliers_title = 'outlier trf:%s, minC_mean: %.2f, maxC_mean: %.2f' % (
      trf_tuple, np.mean(np.min(outlier_image_trf, axis=(0, 1))),
      np.mean(np.max(outlier_image_trf, axis=(0, 1))))
    print('outlier %i, maxC_mean: %s, minC_mean: %s' % (
      outlier_idx, str(np.max(outlier_image_trf, axis=(0, 1))),
      str(np.min(outlier_image_trf, axis=(0, 1)))))
    data_loader.plot_image(inlier_image_trf, show=SHOW_PLOT,
                           title=inliers_title)
    data_loader.plot_image(outlier_image_trf, show=SHOW_PLOT,
                           title=outliers_title)
