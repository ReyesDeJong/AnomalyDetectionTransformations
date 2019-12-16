"""
Selecting transformation by acc matrix
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import tensorflow as tf

from models.transformer_ensemble_ovo_simple_net_od import \
  EnsembleOVOTransformODSimpleModel
import matplotlib.pyplot as plt
import numpy as np
from modules import utils
from models.transform_selectors.selector_by_ovo_discrimination import TransformSelectorByOVO




if __name__ == '__main__':
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  # GETTING MATRICES
  from parameters import loader_keys, general_keys
  from modules.geometric_transform import transformations_tf
  from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
  from models.transformer_od_simple_net import TransformODSimpleModel

  weights_path = os.path.join(
      PROJECT_PATH,
      'results/transform_selection_1/Ensemble_OVO_Transformer_OD_Simple_Model/Kernel_transformer/Ensemble_OVO_Transformer_OD_Simple_Model_20191211-112609',
      # 'results/transform_selection_2/Ensemble_OVO_Transformer_OD_Simple_Model/72_transformer/Ensemble_OVO_Transformer_OD_Simple_Model_20191210-174727',
      # 'results/transform_selection_3/Ensemble_OVO_Transformer_OD_Simple_Model',
      # '72_transformer/Ensemble_OVO_Transformer_OD_Simple_Model_20191211-113030',
      'checkpoints'
  )

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2, 3],#[2],#
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  data_loader = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader.get_outlier_detection_datasets()
  # transformer = transformations_tf.Transformer()
  transformer = transformations_tf.KernelTransformer(
      flips=True, gauss=False, log=False)
  #model
  evaluation_model = TransformODSimpleModel(
      data_loader=data_loader, transformer=transformer,
      input_shape=x_train.shape[1:])
  evaluation_model.fit(x_train, x_val, transform_batch_size=1024)
  met_dict = evaluation_model.evaluate_od(
      x_train, x_test, y_test, 'hits-4-c', 'real', x_val)
  print('\nroc_auc')
  for key in met_dict.keys():
    print(key, met_dict[key]['roc_auc'])
  print('\nacc_at_percentil')
  for key in met_dict.keys():
    print(key, met_dict[key]['acc_at_percentil'])
  print('\nmax_accuracy')
  for key in met_dict.keys():
    print(key, met_dict[key]['max_accuracy'])
  print('\npr_auc_norm')
  for key in met_dict.keys():
    print(key, met_dict[key]['pr_auc_norm'])
  del evaluation_model


  mdl = EnsembleOVOTransformODSimpleModel(
      data_loader=data_loader, transformer=transformer,
      input_shape=x_train.shape[1:])
  mdl.load_model_weights(weights_path)
  mdl.build_models()
  train_acc_matrix = mdl.get_acc_matrix(
      x_train, transform_batch_size=1024, predict_batch_size=2048)
  mdl.plot_score_acc_matrices(train_acc_matrix)
  TransformSelectorByOVO().get_transformer_with_selected_transforms(
      mdl, train_acc_matrix, transformer, verbose=1)
  del mdl
  mdl = EnsembleOVOTransformODSimpleModel(
      data_loader=data_loader, transformer=transformer,
      input_shape=x_train.shape[1:])
  mdl.fit(x_train, x_val, train_batch_size=1024, verbose=0)
  train_acc_matrix = mdl.get_acc_matrix(
      x_train, transform_batch_size=1024, predict_batch_size=2048)
  mdl.plot_score_acc_matrices(train_acc_matrix)
  del mdl

  evaluation_model = TransformODSimpleModel(
      data_loader=data_loader, transformer=transformer,
      input_shape=x_train.shape[1:])
  evaluation_model.fit(x_train, x_val, transform_batch_size=1024)
  met_dict = evaluation_model.evaluate_od(
      x_train, x_test, y_test, 'hits-4-c', 'real', x_val)
  print('\nroc_auc')
  for key in met_dict.keys():
    print(key, met_dict[key]['roc_auc'])
  print('\nacc_at_percentil')
  for key in met_dict.keys():
    print(key, met_dict[key]['acc_at_percentil'])
  print('\nmax_accuracy')
  for key in met_dict.keys():
    print(key, met_dict[key]['max_accuracy'])
  print('\npr_auc_norm')
  for key in met_dict.keys():
    print(key, met_dict[key]['pr_auc_norm'])
  del evaluation_model