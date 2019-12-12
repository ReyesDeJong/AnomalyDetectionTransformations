import os
import sys

"""
Test 72 transform, if transformed data by old transformer is the same in tf2 
and tf1
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import matplotlib;

matplotlib.use('agg')
from modules import utils
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys
import time
import tensorflow as tf
import transformations


def save_normal_and_transformed_data(transformer,
    normal_data_name='tf2_normal.pkl',
    transformed_data_name='tf2_old_transformed.pkl'):
  save_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_data')
  utils.check_paths(save_dir)
  hits_params = {
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
  hits_outlier_dataset = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = hits_outlier_dataset.get_outlier_detection_datasets()
  x_train_transform, y_train_transform = transformer.apply_all_transforms(
      x=x_train)
  x_val_transform, y_val_transform = transformer.apply_all_transforms(
      x=x_val)
  x_test_transform, y_test_transform = transformer.apply_all_transforms(
      x=x_test)
  normal_data = ((x_train, y_train), (x_val, y_val), (x_test, y_test))
  transformed_data = (
    (x_train_transform, y_train_transform), (x_val_transform, y_val_transform),
    (x_test_transform, y_test_transform))
  utils.save_pickle(normal_data, os.path.join(save_dir, normal_data_name))
  utils.save_pickle(transformed_data,
                    os.path.join(save_dir, transformed_data_name))


if __name__ == '__main__':
  results_path = os.path.join(PROJECT_PATH, 'results', 'replication')
  # Not use all gpu
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  # # Transformer
  transformer_old = transformations.Transformer()
  save_normal_and_transformed_data(transformer_old)

