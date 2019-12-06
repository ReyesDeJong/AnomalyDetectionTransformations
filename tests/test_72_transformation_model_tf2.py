import os
import sys

"""
Test 72 transform tf1 model on hits
I mixed this with tf2, sorry
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from modules import utils
import numpy as np
import pandas as pd
import torch
import time
import scipy
from sklearn.metrics import roc_curve, auc
from torch import nn
from models.transformer_od_already_transformed import AlreadyTransformODModel
from models.transformer_od import TransformODModel
from scipy.special import psi, polygamma
from modules.geometric_transform import transformations_tf
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import general_keys, loader_keys
from models.simple_network import create_simple_network
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from models.transformer_od_simple_net import TransformODSimpleModel

N_RUNS = 10


def get_entropy(matrix_scores, epsilon=1e-10):
  norm_scores = matrix_scores / np.sum(matrix_scores, axis=(1, 2))[
    ..., np.newaxis, np.newaxis]
  log_scores = np.log(norm_scores + epsilon)
  product = norm_scores * log_scores
  entropy = -np.sum(product, axis=(1, 2))
  return entropy


def get_xH(transformer, matrix_evals):
  matrix_evals[matrix_evals == 0] += 1e-10
  matrix_evals[matrix_evals == 1] -= 1e-10
  matrix_evals_compĺement = 1 - matrix_evals

  matrix_vals_stack = np.stack([matrix_evals_compĺement, matrix_evals],
                               axis=-1)
  xH = nn.NLLLoss(reduction='none')
  gt_matrix = np.stack(
      [np.eye(transformer.n_transforms)] * len(matrix_vals_stack))
  gt_torch = torch.LongTensor(gt_matrix)
  matrix_logSoftmax_torch = torch.FloatTensor(
      np.swapaxes(np.swapaxes(matrix_vals_stack, 1, -1), -1, -2)).log()
  loss_xH = xH(matrix_logSoftmax_torch, gt_torch)
  batch_xH = np.mean(loss_xH.numpy(), axis=(-1, -2))
  return batch_xH


def calc_approx_alpha_sum(observations):
  N = len(observations)
  f = np.mean(observations, axis=0)

  return (N * (len(f) - 1) * (-psi(1))) / (
      N * np.sum(f * np.log(f)) - np.sum(
      f * np.sum(np.log(observations), axis=0)))


def inv_psi(y, iters=5):
  # initial estimate
  cond = y >= -2.22
  x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / ((y + 1e-10) - psi(1))

  for _ in range(iters):
    x = x - (psi(x) - y) / polygamma(1, x)
  return x


def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
  alpha_new = alpha_old = alpha_init
  for _ in range(max_iter):
    alpha_new = inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
    if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9:
      break
    alpha_old = alpha_new
  return alpha_new


def dirichlet_normality_score(alpha, p):
  return np.sum((alpha - 1) * np.log(p), axis=-1)


def get_roc_auc(scores, labels):
  scores = scores.flatten()
  labels = labels.flatten()

  scores_pos = scores[labels == 1]
  scores_neg = scores[labels != 1]

  truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
  preds = np.concatenate((scores_neg, scores_pos))
  fpr, tpr, roc_thresholds = roc_curve(truth, preds)
  roc_auc = auc(fpr, tpr)
  return roc_auc


def test_tf1_transformed_data_on_tf2_model_original_diri(
    mdl: AlreadyTransformODModel, transformer,
    dataset_name='hits-4-c',
    single_class_ind=1, tf_version='tf1', transformer_name='transtransformed',
    model_name='resnet', epochs=None):
  results_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_results')
  data_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_data')
  utils.check_path(data_dir)
  utils.check_path(os.path.join(results_dir, dataset_name))

  # load data
  normal_data_path = os.path.join(
      data_dir, 'normal_data_%s_%s_loading.pkl' % (dataset_name, tf_version))
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = pd.read_pickle(normal_data_path)

  # load transformed data
  transformed_data_path = os.path.join(
      data_dir,
      '%s_data_%s_%s_loading.pkl' % (
        transformer_name, dataset_name, tf_version))
  (x_train_transform_tf1, y_train_transform_tf1), (
    x_val_transform_tf1, y_val_transform_tf1), (
    x_test_transform_tf1, y_test_transform_tf1) = pd.read_pickle(
      transformed_data_path)
  print(x_train.shape)
  print(x_train_transform_tf1.shape)
  print(x_test.shape)
  print(x_test_transform_tf1.shape)

  # train model
  batch_size = 128
  if epochs is None:
    epochs = int(np.ceil(200 / transformer.n_transforms))
  mdl.fit((x_train_transform_tf1, y_train_transform_tf1),
          (x_val_transform_tf1, y_val_transform_tf1),
          train_batch_size=batch_size, epochs=epochs)
  scores = np.zeros((len(x_test),))
  matrix_evals = np.zeros(
      (len(x_test), transformer.n_transforms, transformer.n_transforms))
  x_pred_train = mdl.predict(x_train_transform_tf1, batch_size=1024)
  x_pred_test = mdl.predict(x_test_transform_tf1, batch_size=1024)
  print(x_pred_train.shape)
  print(x_pred_test.shape)
  for t_ind in range(transformer.n_transforms):
    ind_x_pred_equal_to_t_ind = np.where(y_train_transform_tf1 == t_ind)[0]
    observed_dirichlet = x_pred_train[ind_x_pred_equal_to_t_ind]
    log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

    alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
    alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

    mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
    ind_x_pred_test_equal_to_t_ind = np.where(y_test_transform_tf1 == t_ind)[0]
    x_test_p = x_pred_test[ind_x_pred_test_equal_to_t_ind]
    matrix_evals[:, :, t_ind] += x_test_p
    scores += dirichlet_normality_score(mle_alpha_t, x_test_p)

  scores /= transformer.n_transforms
  matrix_evals /= transformer.n_transforms
  scores_simple = np.trace(matrix_evals, axis1=1, axis2=2)
  scores_entropy = -1 * get_entropy(matrix_evals)
  scores_xH = -1 * get_xH(transformer, matrix_evals)
  labels = y_test.flatten() == single_class_ind

  """
  Time test_model_original(transformer, load_hits4c, dataset_name='hits-4-c', tf_version='tf1') 00:04:31.65
  (0.992217, 0.9895665, 0.99131725, 0.989478125)
  (0.99240075, 0.9900822499999999, 0.99215325, 0.9901300000000001)
  """
  return get_roc_auc(scores, labels), get_roc_auc(scores_simple, labels), \
         get_roc_auc(scores_entropy, labels), get_roc_auc(scores_xH, labels)


def test_tf1_transformed_data_on_tf2_keras_model_diri(
    mdl: tf.keras.Model, transformer,
    dataset_name='hits-4-c',
    single_class_ind=1, tf_version='tf1', transformer_name='transtransformed',
    model_name='resnet', epochs=None):
  results_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_results')
  data_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_data')
  utils.check_path(data_dir)
  utils.check_path(os.path.join(results_dir, dataset_name))

  # load data
  normal_data_path = os.path.join(
      data_dir, 'normal_data_%s_%s_loading.pkl' % (dataset_name, tf_version))
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = pd.read_pickle(normal_data_path)

  # load transformed data
  transformed_data_path = os.path.join(
      data_dir,
      '%s_data_%s_%s_loading.pkl' % (
        transformer_name, dataset_name, tf_version))
  (x_train_transform_tf1, y_train_transform_tf1), (
    x_val_transform_tf1, y_val_transform_tf1), (
    x_test_transform_tf1, y_test_transform_tf1) = pd.read_pickle(
      transformed_data_path)
  print(x_train.shape)
  print(x_train_transform_tf1.shape)
  print(x_test.shape)
  print(x_test_transform_tf1.shape)

  # train model
  batch_size = 128
  if epochs is None:
    epochs = int(np.ceil(200 / transformer.n_transforms))
  mdl.fit(x_train_transform_tf1, to_categorical(y_train_transform_tf1),
          validation_data=(
          x_val_transform_tf1, to_categorical(y_val_transform_tf1)),
          batch_size=batch_size, epochs=epochs)
  scores = np.zeros((len(x_test),))
  matrix_evals = np.zeros(
      (len(x_test), transformer.n_transforms, transformer.n_transforms))
  x_pred_train = mdl.predict(x_train_transform_tf1, batch_size=1024)
  x_pred_test = mdl.predict(x_test_transform_tf1, batch_size=1024)
  print(x_pred_train.shape)
  print(x_pred_test.shape)
  for t_ind in range(transformer.n_transforms):
    ind_x_pred_equal_to_t_ind = np.where(y_train_transform_tf1 == t_ind)[0]
    observed_dirichlet = x_pred_train[ind_x_pred_equal_to_t_ind]
    log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

    alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
    alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

    mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
    ind_x_pred_test_equal_to_t_ind = np.where(y_test_transform_tf1 == t_ind)[0]
    x_test_p = x_pred_test[ind_x_pred_test_equal_to_t_ind]
    matrix_evals[:, :, t_ind] += x_test_p
    scores += dirichlet_normality_score(mle_alpha_t, x_test_p)

  scores /= transformer.n_transforms
  matrix_evals /= transformer.n_transforms
  scores_simple = np.trace(matrix_evals, axis1=1, axis2=2)
  scores_entropy = -1 * get_entropy(matrix_evals)
  scores_xH = -1 * get_xH(transformer, matrix_evals)
  labels = y_test.flatten() == single_class_ind

  """
  Time test_model_original(transformer, load_hits4c, dataset_name='hits-4-c', tf_version='tf1') 00:04:31.65
  (0.992217, 0.9895665, 0.99131725, 0.989478125)
  (0.99240075, 0.9900822499999999, 0.99215325, 0.9901300000000001)
  """
  return get_roc_auc(scores, labels), get_roc_auc(scores_simple, labels), \
         get_roc_auc(scores_entropy, labels), get_roc_auc(scores_xH, labels)


def test_tf1_transformed_data_on_tf2_model_original(
    mdl: AlreadyTransformODModel, transformer,
    dataset_name='hits-4-c',
    single_class_ind=1, tf_version='tf1', transformer_name='transtransformed',
    model_name='resnet', epochs=None):
  """Dirichlet as in object"""
  results_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_results')
  data_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_data')
  utils.check_path(data_dir)
  utils.check_path(os.path.join(results_dir, dataset_name))

  # load data
  normal_data_path = os.path.join(
      data_dir, 'normal_data_%s_%s_loading.pkl' % (dataset_name, tf_version))
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = pd.read_pickle(normal_data_path)

  # load transformed data
  transformed_data_path = os.path.join(
      data_dir,
      '%s_data_%s_%s_loading.pkl' % (
        transformer_name, dataset_name, tf_version))
  (x_train_transform_tf1, y_train_transform_tf1), (
    x_val_transform_tf1, y_val_transform_tf1), (
    x_test_transform_tf1, y_test_transform_tf1) = pd.read_pickle(
      transformed_data_path)
  print(x_train.shape)
  print(x_train_transform_tf1.shape)
  print(x_test.shape)
  print(x_test_transform_tf1.shape)

  # train model
  batch_size = 128
  if epochs is None:
    epochs = int(np.ceil(200 / transformer.n_transforms))
  mdl.fit((x_train_transform_tf1, y_train_transform_tf1),
          (x_val_transform_tf1, y_val_transform_tf1),
          train_batch_size=batch_size, epochs=epochs)
  met_dict = mdl.evaluate_od(
      (x_train_transform_tf1, y_train_transform_tf1),
      (x_test_transform_tf1, y_test_transform_tf1), y_test, dataset_name,
      'real',
      (x_val_transform_tf1, y_val_transform_tf1))

  """
  roc_auc
  dirichlet 0.9896582500000001
  matrix_trace 0.9541035
  entropy 0.9820515000000001
  cross_entropy 0.9614397499999999
  mutual_information 0.9889197499999999
  """
  return met_dict[general_keys.DIRICHLET]['roc_auc'], \
         met_dict[general_keys.MATRIX_TRACE]['roc_auc'], \
         met_dict[general_keys.ENTROPY]['roc_auc'], \
         met_dict[general_keys.CROSS_ENTROPY]['roc_auc']


def test_tf1_normal_data_on_tf2_transformer_model_original(
    mdl: TransformODModel, transformer,
    dataset_name='hits-4-c', tf_version='tf1', epochs=None):
  """Dirichlet as in object"""
  results_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_results')
  data_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_data')
  utils.check_path(data_dir)
  utils.check_path(os.path.join(results_dir, dataset_name))

  # load data
  normal_data_path = os.path.join(
      data_dir, 'normal_data_%s_%s_loading.pkl' % (dataset_name, tf_version))
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = pd.read_pickle(normal_data_path)
  x_train = x_train[y_train == 1]
  x_val = x_val[y_val == 1]

  # train model
  batch_size = 128
  if epochs is None:
    epochs = int(np.ceil(200 / transformer.n_transforms))
  mdl.fit(x_train, x_val, train_batch_size=batch_size, epochs=epochs)
  met_dict = mdl.evaluate_od(
      x_train, x_test, y_test, dataset_name, 'real', x_val)

  return met_dict[general_keys.DIRICHLET]['roc_auc'], \
         met_dict[general_keys.MATRIX_TRACE]['roc_auc'], \
         met_dict[general_keys.ENTROPY]['roc_auc'], \
         met_dict[general_keys.CROSS_ENTROPY]['roc_auc']


def test_all_tf2(
    mdl: TransformODModel, transformer, data_loader,
    dataset_name='hits-4-c', epochs=None):
  """Dirichlet as in object"""
  results_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_results')
  data_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_data')
  utils.check_path(data_dir)
  utils.check_path(os.path.join(results_dir, dataset_name))

  # load data
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader.get_outlier_detection_datasets()

  # train model
  batch_size = 128
  if epochs is None:
    epochs = int(np.ceil(200 / transformer.n_transforms))
  mdl.fit(x_train, x_val, train_batch_size=batch_size, epochs=epochs)
  met_dict = mdl.evaluate_od(
      x_train, x_test, y_test, dataset_name, 'real', x_val)

  return met_dict[general_keys.DIRICHLET]['roc_auc'], \
         met_dict[general_keys.MATRIX_TRACE]['roc_auc'], \
         met_dict[general_keys.ENTROPY]['roc_auc'], \
         met_dict[general_keys.CROSS_ENTROPY]['roc_auc']


def group_scores(scores_list):
  n_scores = len(scores_list[0])
  grouped_scores_list = []
  for score_i in range(n_scores):
    score_i_list = []
    for score_i_it_j in range(len(scores_list)):
      score_i_list.append(scores_list[score_i_it_j][score_i])
    grouped_scores_list.append(score_i_list)
  return grouped_scores_list


def print_scores_times_to_file(file_path, header, scores_list, times_list):
  with open(file_path, "a+") as text_file:
    text_file.write(header + '\n')
    scores_grouped = group_scores(scores_list)
    metric_names = ['diri', 'simple', 'entropy', 'xH']
    for i in range(len(scores_grouped)):
      text_file.write('%s: %.5f+/-%.5f\n' % (metric_names[i],
                                             np.mean(scores_grouped[i]),
                                             scipy.stats.sem(
                                                 scores_grouped[i])))
      text_file.write("Time: %s\n\n" % utils.delta_timer(np.mean(times_list)))


###########################TEsts

def test_resnet_transformer():
  transformer = transformations_tf.Transformer()
  scores_list = []
  delta_times_list = []
  for i in range(N_RUNS):
    mdl = AlreadyTransformODModel(
        transformer=transformer,
        input_shape=[21, 21, 4])
    start_time = time.time()
    scores = test_tf1_transformed_data_on_tf2_model_original_diri(
        mdl, transformer, dataset_name='hits-4-c', tf_version='tf1',
        transformer_name='transformed', model_name='resnet')
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
    del mdl
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_transformer_tf1_models_diri_tf2_resnet_transformer\n NRUNS: %i' % N_RUNS,
                             scores_list, delta_times_list)


def test_resnet_transtransformer_tf2_unchanged():
  transformer = transformations_tf.TransTransformer()
  scores_list = []
  delta_times_list = []
  for i in range(N_RUNS):
    mdl = AlreadyTransformODModel(
        transformer=transformer,
        input_shape=(21, 21, 4))
    start_time = time.time()
    scores = test_tf1_transformed_data_on_tf2_model_original(
        mdl, transformer, dataset_name='hits-4-c', tf_version='tf1',
        transformer_name='transtransformed', model_name='resnet', epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
    del mdl
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_transformer_tf1_models_tf2_unchanged_resnet_transtransformer\n NRUNS: %i' % N_RUNS,
                             scores_list, delta_times_list)


def test_tf2_resnet_transtransformer_unchanged():
  transformer = transformations_tf.TransTransformer()
  scores_list = []
  delta_times_list = []
  for i in range(N_RUNS):
    mdl = TransformODModel(data_loader=None,
                           transformer=transformer,
                           input_shape=(21, 21, 4))
    start_time = time.time()
    scores = test_tf1_normal_data_on_tf2_transformer_model_original(
        mdl, transformer, dataset_name='hits-4-c', tf_version='tf1', epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
    del mdl
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_normal_tf1_models_and_transforms_tf2_unchanged_resnet_transtransformer\n NRUNS: %i' % N_RUNS,
                             scores_list, delta_times_list)


def test_tf2_resnet_transtransformer_unchanged_1c():
  transformer = transformations_tf.TransTransformer()
  scores_list = []
  delta_times_list = []
  for i in range(N_RUNS):
    mdl = TransformODModel(data_loader=None,
                           transformer=transformer,
                           input_shape=(21, 21, 1))
    start_time = time.time()
    scores = test_tf1_normal_data_on_tf2_transformer_model_original(
        mdl, transformer, dataset_name='hits-1-c', tf_version='tf1', epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
    del mdl
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_normal_tf1_models_and_transforms_tf2_unchanged_resnet_transtransformer_1c\n NRUNS: %i' % N_RUNS,
                             scores_list, delta_times_list)


def test_resnet_transtransformer_tf2_unchanged_1c():
  transformer = transformations_tf.TransTransformer()
  scores_list = []
  delta_times_list = []
  for i in range(N_RUNS):
    mdl = AlreadyTransformODModel(
        transformer=transformer,
        input_shape=(21, 21, 1))
    start_time = time.time()
    scores = test_tf1_transformed_data_on_tf2_model_original(
        mdl, transformer, dataset_name='hits-1-c', tf_version='tf1',
        transformer_name='transtransformed', model_name='resnet', epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
    del mdl
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_transformer_tf1_models_tf2_unchanged_resnet_transtransformer_1c\n NRUNS: %i' % N_RUNS,
                             scores_list, delta_times_list)


def test_resnet_transtransformer():
  transformer = transformations_tf.TransTransformer()
  scores_list = []
  delta_times_list = []
  for i in range(N_RUNS):
    mdl = AlreadyTransformODModel(
        transformer=transformer,
        input_shape=(21, 21, 4))
    start_time = time.time()
    scores = test_tf1_transformed_data_on_tf2_model_original_diri(
        mdl, transformer, dataset_name='hits-4-c', tf_version='tf1',
        transformer_name='transtransformed', model_name='resnet', epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
    del mdl
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_transformer_tf1_models_diri_tf2_resnet_transtransformer\n NRUNS: %i' % N_RUNS,
                             scores_list, delta_times_list)


def test_all_tf2_resnet_transtransformer():
  transformer = transformations_tf.TransTransformer()
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
  scores_list = []
  delta_times_list = []
  for i in range(N_RUNS):
    mdl = TransformODModel(
        data_loader=None, transformer=transformer, input_shape=(21, 21, 4))
    start_time = time.time()
    scores = test_all_tf2(
        mdl, transformer, hits_outlier_dataset, dataset_name='hits-4-c',
        epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
    del mdl
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'all_tf2_unchanged_resnet_transtransformer\n NRUNS: %i' % N_RUNS,
                             scores_list, delta_times_list)


def test_all_tf2_resnet_transtransformer_1c():
  transformer = transformations_tf.TransTransformer()
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
  hits_outlier_dataset = HiTSOutlierLoader(hits_params)
  scores_list = []
  delta_times_list = []
  for i in range(N_RUNS):
    mdl = TransformODModel(
        data_loader=None, transformer=transformer, input_shape=(21, 21, 1))
    start_time = time.time()
    scores = test_all_tf2(
        mdl, transformer, hits_outlier_dataset, dataset_name='hits-1-c',
        epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
    del mdl
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'all_tf2_unchanged_resnet_transtransformer_1c\n NRUNS: %i' % N_RUNS,
                             scores_list, delta_times_list)


def test_dh_transtransformer_1c():
  transformer = transformations_tf.TransTransformer()
  scores_list = []
  delta_times_list = []
  for i in range(N_RUNS):
    mdl = create_simple_network(
        input_shape=(21, 21, 1), num_classes=transformer.n_transforms,
        dropout_rate=0.0)
    mdl.compile('adam', 'categorical_crossentropy', ['acc'])
    start_time = time.time()
    scores = test_tf1_transformed_data_on_tf2_keras_model_diri(
        mdl, transformer, dataset_name='hits-1-c', tf_version='tf1',
        transformer_name='transtransformed', model_name='dh', epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
    del mdl
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_transformer_tf1_models_diri_tf2_dh_transtransformer_functionModel\n NRUNS: %i' % N_RUNS,
                             scores_list, delta_times_list)

def test_all_tf2_dh_transtransformer_1c():
  transformer = transformations_tf.TransTransformer()
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
  hits_outlier_dataset = HiTSOutlierLoader(hits_params)
  scores_list = []
  delta_times_list = []
  for i in range(N_RUNS):
    mdl = TransformODSimpleModel(
        data_loader=None, transformer=transformer, input_shape=(21, 21, 1), drop_rate=0.0)
    start_time = time.time()
    scores = test_all_tf2(
        mdl, transformer, hits_outlier_dataset, dataset_name='hits-1-c',
        epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
    del mdl
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'all_tf2_unchanged_dh_transtransformer_1c_DP0.0_fast_no_prints\n NRUNS: %i' % N_RUNS,
                             scores_list, delta_times_list)


if __name__ == '__main__':
  # transformer = transformations.Transformer()
  # start_time = time.time()
  # scores = test_model_original(transformer, load_hits4c,
  #                              dataset_name='hits-4-c')
  # print(
  #     "Time test_model_original(transformer, load_hits4c, dataset_name='hits-4-c') %s" % utils.timer(
  #         start_time, time.time()),
  #     flush=True)
  # print(scores)
  # test_tf2_resnet_transtransformer_unchanged()
  # test_dh_transtransformer_1c()
  test_all_tf2_dh_transtransformer_1c()