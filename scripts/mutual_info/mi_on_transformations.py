import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys

"""
See if chinos MI is compatible with TF2
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from modules.info_metrics.information_estimator_v2 import InformationEstimator
from modules.info_metrics.gaussian_dataset import generate_dataset
from trainers.base_trainer import Trainer
from parameters import loader_keys, general_keys, param_keys
from models.transformer_od import TransformODModel
from modules.geometric_transform import transformations_tf
import tensorflow as tf
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
# from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
import time
from modules.utils import timer

if __name__ == '__main__':
  transformations_arg_to_use_from_available = 0
  random_seed = 42
  dimension = 10
  n_samples_batch = 128
  sigma_zero = 2.0
  corr_factor = 0.9
  x_samples, y_samples, real_MI = generate_dataset(dimension, corr_factor,
                                                   n_samples_batch)
  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2, 3],  # [2],#
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  data_loader = HiTSOutlierLoader(hits_params)
  _, (x_val, y_val), _ = data_loader.get_outlier_detection_datasets()
  x_samples = x_val

  transformer_72 = transformations_tf.Transformer()
  print(transformer_72.transformation_tuples)
  rotation_90_arg = transformer_72.transformation_tuples.index((False, 0, 0, 1))
  trans_down_arg = transformer_72.transformation_tuples.index((False, 0, -8, 0))
  non_transform_arg = transformer_72.transformation_tuples.index(
      (False, 0, -8, 0))
  transformation_args_list = [non_transform_arg, rotation_90_arg,
                              trans_down_arg]

  transformation_arg = transformation_args_list[transformations_arg_to_use_from_available]
  x_transformed, y_transformed = transformer_72.apply_transforms(
      x_samples, transformation_arg)
  print(np.unique(y_transformed, return_counts=True))
  rndm_indxs = np.random.RandomState(42).permutation(len(x_samples))
  x_transformed = x_transformed[rndm_indxs]
  y_transformed = y_transformed[rndm_indxs]

  estimation_ds = tf.data.Dataset.from_tensor_slices(
      (x_samples, x_transformed)).batch(n_samples_batch)
  #
  #
  # Estimation new rule
  estimator = InformationEstimator(sigma_zero)
  # mi_list = []
  mean_mi = tf.keras.metrics.Mean(name='MI')
  start_time = time.time()
  for x_orig, x_trans in estimation_ds:
    mi_estimation = estimator.mutual_information(x_samples, y_samples)
    mean_mi(mi_estimation)
  finish_time = time.time()
  print('Comparing original and %s, MI: %f' % (
  transformer_72.transformation_tuples[transformation_arg], mean_mi.result()))
  print(timer(start_time, finish_time))
