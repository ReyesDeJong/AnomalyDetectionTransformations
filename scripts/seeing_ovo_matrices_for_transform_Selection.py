"""
Seeing if transforms are discardable in matrix space
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import tensorflow as tf

from parameters import loader_keys, general_keys
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from models.transformer_ensemble_ovo_simple_net_od import \
  EnsembleOVOTransformODSimpleModel
from modules.geometric_transform import transformations_tf
import matplotlib.pyplot as plt
import numpy as np
from modules import utils


def plot_n_matrices(matrix_scores, N_to_plot):
  for i in range(N_to_plot):
    index = np.random.randint(len(matrix_scores))
    plt.imshow(matrix_scores[index])
    plt.show()


def hits4c_tr18():
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
  data_loader = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader.get_outlier_detection_datasets()
  transformer = transformations_tf.KernelTransformer(
      flips=True, gauss=False, log=False)
  mdl = EnsembleOVOTransformODSimpleModel(
      data_loader, transformer=transformer, input_shape=x_train.shape[1:],
      results_folder_name='transform_selection_1')
  mdl.fit(x_train, x_val, train_batch_size=1024, verbose=0)
  # train_matrix_scores = mdl.predict_matrix_score(
  #     x_train, transform_batch_size=1024)
  # val_matrix_scores = mdl.predict_matrix_score(
  #     x_val, transform_batch_size=1024)
  # test_outlier_matrix_scores = mdl.predict_matrix_score(
  #     x_test[y_test == 0], transform_batch_size=1024)
  # utils.save_pickle(train_matrix_scores, os.path.join(
  #     mdl.main_model_path, 'train_matrix_scores_translations+flip(18)_train_step.pkl'))
  # utils.save_pickle(val_matrix_scores, os.path.join(
  #     mdl.main_model_path, 'val_matrix_scores_translations+flip(18)_train_step.pkl'))
  # utils.save_pickle(test_outlier_matrix_scores, os.path.join(
  #     mdl.main_model_path, 'test_matrix_scores_translations+flip(18)_train_step.pkl'))


def hits4c():
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
  data_loader = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader.get_outlier_detection_datasets()
  # transformer = transformations_tf.KernelTransformer(
  #     flips=True, gauss=False, log=False)
  transformer = transformations_tf.Transformer()
  mdl = EnsembleOVOTransformODSimpleModel(
      data_loader, transformer=transformer, input_shape=x_train.shape[1:],
      results_folder_name='transform_selection_2')
  mdl.fit(x_train, x_val, train_batch_size=1024, verbose=0)
  # train_matrix_scores = mdl.predict_matrix_score(
  #     x_train, transform_batch_size=1024)
  # val_matrix_scores = mdl.predict_matrix_score(
  #     x_val, transform_batch_size=1024)
  # test_outlier_matrix_scores = mdl.predict_matrix_score(
  #     x_test[y_test == 0], transform_batch_size=1024)
  # utils.save_pickle(train_matrix_scores, os.path.join(
  #     mdl.main_model_path, 'train_matrix_scores_72.pkl'))
  # utils.save_pickle(val_matrix_scores, os.path.join(
  #     mdl.main_model_path, 'val_matrix_scores_72.pkl'))
  # utils.save_pickle(test_outlier_matrix_scores, os.path.join(
  #     mdl.main_model_path, 'test_matrix_scores_72.pkl'))


def hits1c():
  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [2],
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  data_loader = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader.get_outlier_detection_datasets()
  # transformer = transformations_tf.KernelTransformer(
  #     flips=True, gauss=False, log=False)
  transformer = transformations_tf.Transformer()
  mdl = EnsembleOVOTransformODSimpleModel(
      data_loader, transformer=transformer, input_shape=x_train.shape[1:],
      results_folder_name='transform_selection_3')
  mdl.fit(x_train, x_val, train_batch_size=1024, verbose=0)
  # train_matrix_scores = mdl.predict_matrix_score(
  #     x_train, transform_batch_size=1024)
  # val_matrix_scores = mdl.predict_matrix_score(
  #     x_val, transform_batch_size=1024)
  # test_outlier_matrix_scores = mdl.predict_matrix_score(
  #     x_test[y_test == 0], transform_batch_size=1024)
  # utils.save_pickle(train_matrix_scores, os.path.join(
  #     mdl.main_model_path, 'train_matrix_scores_72_1c.pkl'))
  # utils.save_pickle(val_matrix_scores, os.path.join(
  #     mdl.main_model_path, 'val_matrix_scores_72_1c.pkl'))
  # utils.save_pickle(test_outlier_matrix_scores, os.path.join(
  #     mdl.main_model_path, 'test_matrix_scores_72_1c.pkl'))


if __name__ == '__main__':
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  hits1c()
  hits4c()
  hits4c_tr18()
