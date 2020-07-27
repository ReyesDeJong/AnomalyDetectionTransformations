"""
Seeing if transforms are discardable in matrix space
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)
import tensorflow as tf

from models.transformer_ensemble_ovo_simple_net_od import \
  EnsembleOVOTransformODSimpleModel
import matplotlib.pyplot as plt
import numpy as np
from modules import utils
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.geometric_transform.transformations_tf import AbstractTransformer
from models.transform_selectors.selector_by_ovo_discrimination import \
  TransformSelectorByOVO
import copy
import pandas as pd

TRANSFORM_SELECTION_RESULTS_FOLDER = os.path.join(PROJECT_PATH, 'results',
                                                  'transform_selection_reports')
TRANSFORM_SELECTION_RESULTS_FILENAME = 'discrepancies_between_val_train_transform_selection.txt'


def try_to_load_model_weights(model: EnsembleOVOTransformODSimpleModel, path=None):
  if path is None:
    path = model.common_to_all_models_transform_selection_checkpoints_folder
  try:

    model.load_model_weights(path)
    return True
  except Exception as e:
    print(e)
    return False


def check_for_discrepancies_between_train_and_val_acc_matrix(
    train_acc_matric, val_acc_matrix,
    data_loader: HiTSOutlierLoader,
    model: EnsembleOVOTransformODSimpleModel,
    transformer: AbstractTransformer, accuracy_selection_tolerance=0.01):
  utils.check_path(TRANSFORM_SELECTION_RESULTS_FOLDER)
  transform_selector = TransformSelectorByOVO()
  train_transformer = copy.deepcopy(transformer)
  val_transformer = copy.deepcopy(transformer)
  transform_selector.get_transformer_with_selected_transforms(
      model, train_acc_matric, train_transformer, verbose=1,
      selection_accuracy_tolerance=accuracy_selection_tolerance)
  transform_selector.get_transformer_with_selected_transforms(
      model, val_acc_matrix, val_transformer, verbose=1,
      selection_accuracy_tolerance=accuracy_selection_tolerance)
  transform_selection_results_path = os.path.join(
    TRANSFORM_SELECTION_RESULTS_FOLDER, TRANSFORM_SELECTION_RESULTS_FILENAME)
  if val_transformer.transformation_tuples == train_transformer.transformation_tuples:
    with open(transform_selection_results_path, "a") as myfile:
      myfile.write(
          "\n" + model.name + '\n' + transformer.name + '\n' + data_loader.name)
      myfile.write(
          'Train and Val Transforms selected are equal')
  else:
    print('DISCREPANCY OF CONFLICTING TRANSFORMS VAL-TRAIN')
    with open(transform_selection_results_path, "a") as myfile:
      myfile.write(
          "\n" + model.name + '\n' + transformer.name + '\n' + data_loader.name)
      myfile.write(
          '[DISCREPANCY] Train and Val Transforms selected are NOT equal')
      myfile.write('Train:\n' + str(train_transformer.transformation_tuples))
      myfile.write('VAL:\n' + str(val_transformer.transformation_tuples))





def get_transform_selection_transformer(data_loader: HiTSOutlierLoader,
    model: EnsembleOVOTransformODSimpleModel,
    transformer: AbstractTransformer, accuracy_selection_tolerance=0.01,
    acc_matrix_to_use_name='val', from_scratch=False):
  transform_selector = TransformSelectorByOVO()
  acc_matrix_index_dict = {'train': 0, 'val': 1}
  acc_matrix_path = os.path.join(
      model.common_to_all_models_transform_selection_results_folder,
      '%s_acc_matrix.pkl' % acc_matrix_to_use_name)
  if os.path.exists(acc_matrix_path) and not from_scratch:
    acc_matrix = pd.read_pickle(acc_matrix_path)
  else:
    both_acc_matrices = generate_transform_selector_acc_matrices(
        data_loader, model, transformer, accuracy_selection_tolerance)
    acc_matrix = both_acc_matrices[
      acc_matrix_index_dict[acc_matrix_to_use_name]]
  transformer_selected = transform_selector.get_transformer_with_selected_transforms(
      model, acc_matrix, transformer, accuracy_selection_tolerance)
  return transformer_selected

def get_acc_matrix(data_loader: HiTSOutlierLoader,
    model: EnsembleOVOTransformODSimpleModel,
    transformer: AbstractTransformer, accuracy_selection_tolerance=0.01,
    acc_matrix_to_use_name='val'):
  acc_matrix_index_dict = {'train': 0, 'val': 1}
  acc_matrix_path = os.path.join(
      model.common_to_all_models_transform_selection_results_folder,
      '%s_acc_matrix.pkl' % acc_matrix_to_use_name)
  if os.path.exists(acc_matrix_path):
    acc_matrix = pd.read_pickle(acc_matrix_path)
  else:
    both_acc_matrices = generate_transform_selector_acc_matrices(
        data_loader, model, transformer, accuracy_selection_tolerance)
    acc_matrix = both_acc_matrices[
      acc_matrix_index_dict[acc_matrix_to_use_name]]
  return acc_matrix


def plot_n_matrices(matrix_scores, N_to_plot):
  for i in range(N_to_plot):
    index = np.random.randint(len(matrix_scores))
    plt.imshow(matrix_scores[index])
    plt.show()


if __name__ == '__main__':
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  # GETTING MATRICES
  from parameters import loader_keys, general_keys
  from modules.geometric_transform import transformations_tf
  from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
  from models.transformer_od_simple_net import TransformODSimpleModel

  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [2],  # [0, 1, 2, 3],#
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  data_loader = HiTSOutlierLoader(hits_params)
  transformer = transformations_tf.KernelTransformer(
      flips=True, gauss=False, log=False)
  print(transformer.n_transforms)

  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader.get_outlier_detection_datasets()
  x_train_shape = x_train.shape[1:]
  del x_train, x_val, x_test, y_train, y_val, y_test
  mdl = EnsembleOVOTransformODSimpleModel(
      data_loader=data_loader, transformer=transformer,
      input_shape=x_train_shape)
  transformer = get_transform_selection_transformer(data_loader, mdl,
                                                    transformer)
  del mdl
  print(transformer.tranformation_to_perform)
  print(transformer.n_transforms)
