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


def try_to_load_model_weights(model: EnsembleOVOTransformODSimpleModel):
  try:
    model.load_model_weights(
        model.common_to_all_models_transform_selection_checkpoints_folder)
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
  if val_transformer.tranformation_to_perform == train_transformer.tranformation_to_perform:
    with open(TRANSFORM_SELECTION_RESULTS_FILENAME, "a") as myfile:
      myfile.write(
          "\n" + model.name + '\n' + transformer.name + '\n' + data_loader.name)
      myfile.write(
          'Train and Val Transforms selected are equal')
  else:
    with open(TRANSFORM_SELECTION_RESULTS_FILENAME, "a") as myfile:
      myfile.write(
          "\n" + model.name + '\n' + transformer.name + '\n' + data_loader.name)
      myfile.write(
          '[DISCREPANCY] Train and Val Transforms selected are NOT equal')
      myfile.write('Train:\n' + str(train_transformer.tranformation_to_perform))
      myfile.write('VAL:\n' + str(val_transformer.tranformation_to_perform))


# TODO: move this and above to methods of model
def generate_transform_selector_acc_matrices(data_loader: HiTSOutlierLoader,
    model: EnsembleOVOTransformODSimpleModel,
    transformer: AbstractTransformer, accuracy_selection_tolerance=0.01):
  # transform_selector = TransformSelectorByOVO()
  (x_train, y_train), (x_val, y_val), (
    _, _) = data_loader.get_outlier_detection_datasets()
  is_model_loaded = try_to_load_model_weights(model)
  if is_model_loaded == False:
    model.fit(x_train, x_val, train_batch_size=1024, verbose=0)
  train_acc_matrix = model.get_acc_matrix(
      x_train, transform_batch_size=1024, predict_batch_size=2048)
  val_acc_matrix = model.get_acc_matrix(
      x_val, transform_batch_size=1024, predict_batch_size=2048)
  utils.save_pickle(
      train_acc_matrix,
      os.path.join(
          model.common_to_all_models_transform_selection_results_folder,
          'train_acc_matrix.pkl'))
  utils.save_pickle(
      val_acc_matrix,
      os.path.join(
          model.common_to_all_models_transform_selection_results_folder,
          'val_acc_matrix.pkl'))
  # TODO: save plots
  utils.save_2d_image(
      train_acc_matrix, 'train_acc_matrix',
      model.common_to_all_models_transform_selection_results_folder,
      axis_show='on')
  utils.save_2d_image(
      val_acc_matrix, 'val_acc_matrix',
      model.common_to_all_models_transform_selection_results_folder,
      axis_show='on')
  check_for_discrepancies_between_train_and_val_acc_matrix(
      train_acc_matrix, val_acc_matrix, data_loader, model, transformer,
      accuracy_selection_tolerance)
  return train_acc_matrix, val_acc_matrix


def get_transform_selection_transformer(data_loader: HiTSOutlierLoader,
    model: EnsembleOVOTransformODSimpleModel,
    transformer: AbstractTransformer, accuracy_selection_tolerance=0.01,
    acc_matrix_to_use_name='val'):
  transform_selector = TransformSelectorByOVO()
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
  transformer_selected = transform_selector.get_transformer_with_selected_transforms(
      model, acc_matrix, transformer, accuracy_selection_tolerance)
  return transformer_selected


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
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader.get_outlier_detection_datasets()
  x_train_shape = x_train.shape[1:]
  del x_train, x_val, x_test, y_train, y_val, y_test
  transformer = transformations_tf.KernelTransformer(
      flips=True, gauss=False, log=False)

  mdl = EnsembleOVOTransformODSimpleModel(
      data_loader=data_loader, transformer=transformer,
      input_shape=x_train.shape[1:])

  transformer = get_transform_selection_transformer(data_loader, mdl,
                                                    transformer)
  print(transformer.tranformation_to_perform)
