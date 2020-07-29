import os
import sys

"""
Test 72 transform tf1 model on hits
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from modules import utils
import time
import transformations
import numpy as np
from models.wide_residual_network import create_wide_residual_network
from scipy.special import psi, polygamma
import torch
import torch.nn as nn
from modules.data_loaders.base_line_loaders import save_roc_pr_curve_data, \
  get_class_name_from_index, load_hits4c, load_hits1c
import datetime
from keras.utils import to_categorical
from scripts.ensemble_transform_vs_all_od_hits import get_entropy
from sklearn.metrics import roc_curve, auc
import pandas as pd
import keras.backend as K
import tensorflow as tf
from models.simple_network import create_simple_network
import scipy


def save_results_file(results_dir, dataset_name, single_class_ind, scores,
    labels,
    experiment_name):
  res_file_name = '{}_{}_{}_{}.npz'.format(dataset_name, experiment_name,
                                           get_class_name_from_index(
                                               single_class_ind, dataset_name),
                                           datetime.datetime.now().strftime(
                                               '%Y-%m-%d-%H%M'))
  res_file_path = os.path.join(results_dir, dataset_name, res_file_name)
  save_roc_pr_curve_data(scores, labels, res_file_path)


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


def test_model_original(transformer, loader, dataset_name='hits-4-c',
    single_class_ind=1):
  results_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_results')
  save_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_data')
  utils.check_path(results_dir)
  utils.check_path(save_dir)
  utils.check_path(os.path.join(results_dir, dataset_name))

  # load-save data
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = loader(return_val=True)
  normal_data = (x_train, y_train), (x_val, y_val), (x_test, y_test)
  utils.save_pickle(normal_data, os.path.join(
      save_dir, 'normal_data_%s_tf1_original.pkl' % dataset_name))
  # create model
  n, k = (10, 4)
  mdl = create_wide_residual_network(
      x_train.shape[1:], transformer.n_transforms, n, k)
  mdl.compile('adam', 'categorical_crossentropy', ['acc'])
  # get inliers of specific class
  # get inliers
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  print(x_train_task.shape)
  # transform inliers
  transformations_inds = np.tile(np.arange(transformer.n_transforms),
                                 len(x_train_task))
  x_train_task_transformed = transformer.transform_batch(
      np.repeat(x_train_task, transformer.n_transforms, axis=0),
      transformations_inds)
  print(x_train_task_transformed.shape)
  # train model
  batch_size = 128
  mdl.fit(x_train=x_train_task_transformed, y=to_categorical(transformations_inds),
          batch_size=batch_size,
          epochs=int(np.ceil(200 / transformer.n_transforms))
          )
  scores = np.zeros((len(x_test),))
  matrix_evals = np.zeros(
      (len(x_test), transformer.n_transforms, transformer.n_transforms))
  observed_data = x_train_task
  for t_ind in range(transformer.n_transforms):
    observed_dirichlet = mdl.predict(
        transformer.transform_batch(observed_data,
                                    [t_ind] * len(observed_data)),
        batch_size=1024)
    log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

    alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
    alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

    mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)

    x_test_p = mdl.predict(
        transformer.transform_batch(x_test, [t_ind] * len(x_test)),
        batch_size=1024)
    matrix_evals[:, :, t_ind] += x_test_p
    scores += dirichlet_normality_score(mle_alpha_t, x_test_p)

  scores /= transformer.n_transforms
  matrix_evals /= transformer.n_transforms
  scores_simple = np.trace(matrix_evals, axis1=1, axis2=2)
  scores_entropy = -1 * get_entropy(matrix_evals)
  scores_xH = -1 * get_xH(transformer, matrix_evals)
  labels = y_test.flatten() == single_class_ind

  save_results_file(results_dir, dataset_name, single_class_ind, scores=scores,
                    labels=labels, experiment_name='transformations')
  save_results_file(results_dir, dataset_name, single_class_ind,
                    scores=scores_simple,
                    labels=labels, experiment_name='transformations-simple')
  save_results_file(results_dir, dataset_name, single_class_ind,
                    scores=scores_entropy,
                    labels=labels, experiment_name='transformations-entropy')
  save_results_file(results_dir, dataset_name, single_class_ind,
                    scores=scores_xH,
                    labels=labels, experiment_name='transformations-xH')
  mdl_weights_name = '{}_tf1_original_{}_weights.h5'.format(dataset_name,
                                                            get_class_name_from_index(
                                                                single_class_ind,
                                                                dataset_name))
  mdl_weights_path = os.path.join(results_dir, dataset_name, mdl_weights_name)
  mdl.save_weights(mdl_weights_path)
  """
  Time test_model_original(transformer, load_hits4c, dataset_name='hits-4-c') 00:06:58.37
  (0.9917134999999999, 0.9350055, 0.9872614999999999, 0.94142025)
  (0.9938067500000001, 0.9923547500000001, 0.9931685, 0.992637375)
  (0.9912172499999999, 0.9883357499999998, 0.9909070000000001, 0.9886706249999999)
  #train only Time test_model_original(transformer, load_hits4c, dataset_name='hits-4-c', tf_version='tf1') 00:03:48.29
  """
  return get_roc_auc(scores, labels), get_roc_auc(scores_simple, labels), \
         get_roc_auc(scores_entropy, labels), get_roc_auc(scores_xH, labels)


def test_model_loading(transformer, mdl, loader, dataset_name='hits-4-c',
    single_class_ind=1, tf_version='tf1', transformer_name='transformed',
    model_name='resnet', epochs=None):
  results_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_results')
  save_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_data')
  utils.check_path(results_dir)
  utils.check_path(save_dir)
  utils.check_path(os.path.join(results_dir, dataset_name))

  # load-save data
  normal_data_path = os.path.join(
      save_dir, 'normal_data_%s_%s_loading.pkl' % (dataset_name, tf_version))
  if os.path.exists(normal_data_path):
    (x_train, y_train), (x_val, y_val), (
      x_test, y_test) = pd.read_pickle(normal_data_path)
  else:
    (x_train, y_train), (x_val, y_val), (
      x_test, y_test) = loader(return_val=True)
    normal_data = (x_train, y_train), (x_val, y_val), (x_test, y_test)
    utils.save_pickle(normal_data, normal_data_path)
  # create model
  # n, k = (10, 4)
  # mdl = create_wide_residual_network(
  #     x_train.shape[1:], transformer.n_transforms, n, k)
  mdl.compile('adam', 'categorical_crossentropy', ['acc'])
  # selec inliers
  x_train = x_train[y_train.flatten() == single_class_ind]
  x_val = x_val[y_val.flatten() == single_class_ind]

  # load-save transformed data
  transformed_data_path = os.path.join(
      save_dir,
      '%s_data_%s_%s_loading.pkl' % (
        transformer_name, dataset_name, tf_version))
  if os.path.exists(transformed_data_path):
    (x_train_transform_tf1, y_train_transform_tf1), (
      x_val_transform_tf1, y_val_transform_tf1), (
      x_test_transform_tf1, y_test_transform_tf1) = pd.read_pickle(
        transformed_data_path)
  else:
    # transform all data
    y_train_transform_tf1 = np.tile(np.arange(transformer.n_transforms),
                                    len(x_train))
    x_train_transform_tf1 = transformer.transform_batch(
        np.repeat(x_train, transformer.n_transforms, axis=0),
        y_train_transform_tf1)
    y_val_transform_tf1 = np.tile(np.arange(transformer.n_transforms),
                                  len(x_val))
    x_val_transform_tf1 = transformer.transform_batch(
        np.repeat(x_val, transformer.n_transforms, axis=0),
        y_val_transform_tf1)
    y_test_transform_tf1 = np.tile(np.arange(transformer.n_transforms),
                                   len(x_test))
    x_test_transform_tf1 = transformer.transform_batch(
        np.repeat(x_test, transformer.n_transforms, axis=0),
        y_test_transform_tf1)
    transformed_data = (
      (x_train_transform_tf1, y_train_transform_tf1), (x_val_transform_tf1,
                                                       y_val_transform_tf1),
      (x_test_transform_tf1, y_test_transform_tf1))
    utils.save_pickle(transformed_data, transformed_data_path)
  print(x_train.shape)
  print(x_train_transform_tf1.shape)
  print(x_test.shape)
  print(x_test_transform_tf1.shape)
  # train model
  batch_size = 128
  if epochs is None:
    epochs = int(np.ceil(200 / transformer.n_transforms))
  mdl.fit(x_train=x_train_transform_tf1, y=to_categorical(y_train_transform_tf1),
          batch_size=batch_size,
          epochs=epochs
          )
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

  save_results_file(results_dir, dataset_name, single_class_ind, scores=scores,
                    labels=labels,
                    experiment_name='%s-%s-loading-%s' % (
                      model_name, transformer_name, tf_version))
  save_results_file(results_dir, dataset_name, single_class_ind,
                    scores=scores_simple,
                    labels=labels,
                    experiment_name='%s-%s-simple-loading-%s' % (
                      model_name, transformer_name, tf_version))
  save_results_file(results_dir, dataset_name, single_class_ind,
                    scores=scores_entropy,
                    labels=labels,
                    experiment_name='%s-%s-entropy-loading-%s' % (
                      model_name, transformer_name, tf_version))
  save_results_file(results_dir, dataset_name, single_class_ind,
                    scores=scores_xH,
                    labels=labels,
                    experiment_name='%s-%s-xH-loading-%s' % (
                      model_name, transformer_name, tf_version))
  mdl_weights_name = '{}_{}_{}_{}_loading_{}_weights.h5'.format(model_name,
                                                                transformer_name,
                                                                dataset_name,
                                                                tf_version,
                                                                get_class_name_from_index(
                                                                    single_class_ind,
                                                                    dataset_name))
  mdl_weights_path = os.path.join(results_dir, dataset_name, mdl_weights_name)
  mdl.save_weights(mdl_weights_path)
  reset_weights()
  """
  Time test_model_original(transformer, load_hits4c, dataset_name='hits-4-c', tf_version='tf1') 00:04:31.65
  (0.992217, 0.9895665, 0.99131725, 0.989478125)
  (0.99240075, 0.9900822499999999, 0.99215325, 0.9901300000000001)
  """
  return get_roc_auc(scores, labels), get_roc_auc(scores_simple, labels), \
         get_roc_auc(scores_entropy, labels), get_roc_auc(scores_xH, labels)


def reset_weights():
  K.get_session().run(tf.global_variables_initializer())


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
      text_file.write('%s: %.5f+/-%.5f\n' % ( metric_names[i],
        np.mean(scores_grouped[i]), scipy.stats.sem(scores_grouped[i])))
      text_file.write("Time: %s\n\n" % utils.delta_timer(np.mean(times_list)))


def test_resnet_transformer():
  n_runs = 10
  transformer = transformations.Transformer()
  n, k = (10, 4)
  mdl_resnet = create_wide_residual_network(
      [21, 21, 4], transformer.n_transforms, n, k)
  scores_list = []
  delta_times_list = []
  for i in range(n_runs):
    start_time = time.time()
    scores = test_model_loading(transformer, mdl_resnet, load_hits4c,
                                dataset_name='hits-4-c', tf_version='tf1',
                                transformer_name='transformed',
                                model_name='resnet')
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_transformer_tf1_models_tf1_resnet_transformer\n NRUNS: %i' % n_runs,
                             scores_list, delta_times_list)
  del mdl_resnet

def test_resnet_transformer_1c():
  n_runs = 10
  transformer = transformations.Transformer()
  n, k = (10, 4)
  mdl_resnet = create_wide_residual_network(
      [21, 21, 1], transformer.n_transforms, n, k)
  scores_list = []
  delta_times_list = []
  for i in range(n_runs):
    start_time = time.time()
    scores = test_model_loading(transformer, mdl_resnet, load_hits1c,
                                dataset_name='hits-1-c', tf_version='tf1',
                                transformer_name='transformed',
                                model_name='resnet')
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_transformer_tf1_models_tf1_resnet_transformer_1c\n NRUNS: %i' % n_runs,
                             scores_list, delta_times_list)
  del mdl_resnet

def test_resnet_transtransformer():
  n_runs = 10
  transformer = transformations.TransTransformer()
  n, k = (10, 4)
  mdl_resnet = create_wide_residual_network(
      [21, 21, 4], transformer.n_transforms, n, k)
  scores_list = []
  delta_times_list = []
  for i in range(n_runs):
    start_time = time.time()
    scores = test_model_loading(transformer, mdl_resnet, load_hits4c,
                                dataset_name='hits-4-c', tf_version='tf1',
                                transformer_name='transtransformed',
                                model_name='resnet', epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_transformer_tf1_models_tf1_resnet_transtransformer\n NRUNS: %i' % n_runs,
                             scores_list, delta_times_list)
  del mdl_resnet

def test_resnet_transtransformer_1c():
  n_runs = 10
  transformer = transformations.TransTransformer()
  n, k = (10, 4)
  mdl_resnet = create_wide_residual_network(
      [21, 21, 1], transformer.n_transforms, n, k)
  scores_list = []
  delta_times_list = []
  for i in range(n_runs):
    start_time = time.time()
    scores = test_model_loading(transformer, mdl_resnet, load_hits1c,
                                dataset_name='hits-1-c', tf_version='tf1',
                                transformer_name='transtransformed',
                                model_name='resnet', epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_transformer_tf1_models_tf1_resnet_transtransformer_1c\n NRUNS: %i' % n_runs,
                             scores_list, delta_times_list)
  del mdl_resnet

def test_dh_transtransformer_1c():
  n_runs = 10
  transformer = transformations.TransTransformer()
  mdl = create_simple_network(
      [21, 21, 1], transformer.n_transforms)
  scores_list = []
  delta_times_list = []
  for i in range(n_runs):
    start_time = time.time()
    scores = test_model_loading(transformer, mdl, load_hits1c,
                                dataset_name='hits-1-c', tf_version='tf1',
                                transformer_name='transtransformed',
                                model_name='dh', epochs=2)
    end_time = time.time()
    delta_times_list.append(end_time - start_time)
    scores_list.append(scores)
  file_path = os.path.join(PROJECT_PATH, 'tests', 'aux_results',
                           'test_models_tf1-tf2.txt')
  print_scores_times_to_file(file_path,
                             'Data_transformer_tf1_models_tf1_dh_transtransformer_1c\n NRUNS: %i' % n_runs,
                             scores_list, delta_times_list)
  del mdl

if __name__ == '__main__':
  K.set_session(tf.Session())
  results_path = os.path.join(PROJECT_PATH, 'results', 'replication')

  # transformer = transformations.Transformer()
  # start_time = time.time()
  # scores = test_model_original(transformer, load_hits4c,
  #                              dataset_name='hits-4-c')
  # print(
  #     "Time test_model_original(transformer, load_hits4c, dataset_name='hits-4-c') %s" % utils.timer(
  #         start_time, time.time()),
  #     flush=True)
  # print(scores)

  test_resnet_transtransformer()
  #
  # test_resnet_transtransformer_1c()
  # test_dh_transtransformer_1c()

