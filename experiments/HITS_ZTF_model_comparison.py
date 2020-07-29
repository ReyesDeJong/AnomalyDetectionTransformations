import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

import csv
from collections import defaultdict
from glob import glob
# from datetime import datetime
from multiprocessing import Manager, freeze_support, Process
import numpy as np
import scipy.stats
from scipy.special import psi, polygamma
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed
from keras.models import Model, Sequential
from keras import Input
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from modules.data_loaders.base_line_loaders import save_roc_pr_curve_data, \
  get_class_name_from_index, get_channels_axis, load_ztf_real_bog, load_hits1c, \
  load_hits4c
from modules.data_loaders import base_line_loaders
from utils import save_roc_pr_curve_data, get_channels_axis
from transformations import Transformer, TransTransformer, KernelTransformer, \
  PlusKernelTransformer
from models.wide_residual_network import create_wide_residual_network
from models.encoders_decoders import conv_encoder, conv_decoder
from models import dsebm, dagmm, adgan
from modules.utils import check_paths
import time
import datetime
from pyod.models.mo_gaal import MO_GAAL
from scripts.ensemble_transform_vs_all_od_hits import get_entropy
import torch
import torch.nn as nn
from modules.utils import check_path

RESULTS_DIR = os.path.join(PROJECT_PATH, 'results/MODEL_COMPARISON')
LARGE_DATASET_NAMES = ['cats-vs-dogs', 'hits', 'hits_padded']
PARALLEL_N_JOBS = 7


def save_results_file(dataset_name, single_class_ind, scores, labels,
    experiment_name):
  res_file_name = '{}_{}_{}_{}.npz'.format(dataset_name, experiment_name,
                                           get_class_name_from_index(
                                               single_class_ind, dataset_name),
                                           datetime.datetime.now().strftime(
                                               '%Y-%m-%d-%H%M'))
  res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
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
  x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / (y - psi(1))

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


def _transformations_experiment(dataset_load_fn, dataset_name, single_class_ind,
    gpu_q):
  # gpu_to_use = gpu_q.get()
  # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

  (x_train, y_train), (x_test, y_test) = dataset_load_fn()

  if dataset_name in ['cats-vs-dogs']:
    transformer = Transformer(16, 16)
    n, k = (16, 8)
  else:
    transformer = Transformer(8, 8)
    n, k = (10, 4)
  mdl = create_wide_residual_network(x_train.shape[1:],
                                     transformer.n_transforms, n, k)
  mdl.compile('adam',
              'categorical_crossentropy',
              ['acc'])

  # get inliers of specific class
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  # [0_i, ..., (N_transforms-1)_i, ..., ..., 0_N_samples, ...,
  # (N_transforms-1)_N_samples] shape: (N_transforms*N_samples,)
  transformations_inds = np.tile(np.arange(transformer.n_transforms),
                                 len(x_train_task))
  x_train_task_transformed = transformer.transform_batch(
      np.repeat(x_train_task, transformer.n_transforms, axis=0),
      transformations_inds)
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
  scores_entropy = get_entropy(matrix_evals)
  scores_xH = get_xH(transformer, matrix_evals)
  labels = y_test.flatten() == single_class_ind

  save_results_file(dataset_name, single_class_ind, scores=scores,
                    labels=labels, experiment_name='transformations')
  save_results_file(dataset_name, single_class_ind, scores=scores_simple,
                    labels=labels, experiment_name='transformations-simple')
  save_results_file(dataset_name, single_class_ind, scores=scores_entropy,
                    labels=labels, experiment_name='transformations-entropy')
  save_results_file(dataset_name, single_class_ind, scores=scores_xH,
                    labels=labels, experiment_name='transformations-xH')

  mdl_weights_name = '{}_transformations_{}_{}_weights.h5'.format(dataset_name,
                                                                  get_class_name_from_index(
                                                                      single_class_ind,
                                                                      dataset_name),
                                                                  datetime.datetime.now().strftime(
                                                                      '%Y-%m-%d-%H%M'))
  mdl_weights_path = os.path.join(RESULTS_DIR, dataset_name, mdl_weights_name)
  mdl.save_weights(mdl_weights_path)

  # gpu_q.put(gpu_to_use)


def _trans_transformations_experiment(dataset_load_fn, dataset_name,
    single_class_ind, gpu_q):
  # gpu_to_use = gpu_q.get()
  # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

  (x_train, y_train), (x_test, y_test) = dataset_load_fn()

  if dataset_name in ['cats-vs-dogs']:
    transformer = TransTransformer(16, 16)
    n, k = (16, 8)
  else:
    transformer = TransTransformer(8, 8)
    n, k = (10, 4)
  mdl = create_wide_residual_network(x_train.shape[1:],
                                     transformer.n_transforms, n, k)
  mdl.compile('adam',
              'categorical_crossentropy',
              ['acc'])

  # get inliers of specific class
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  # [0_i, ..., (N_transforms-1)_i, ..., ..., 0_N_samples, ...,
  # (N_transforms-1)_N_samples] shape: (N_transforms*N_samples,)
  transformations_inds = np.tile(np.arange(transformer.n_transforms),
                                 len(x_train_task))
  x_train_task_transformed = transformer.transform_batch(
      np.repeat(x_train_task, transformer.n_transforms, axis=0),
      transformations_inds)
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
  scores_entropy = get_entropy(matrix_evals)
  scores_xH = get_xH(transformer, matrix_evals)
  labels = y_test.flatten() == single_class_ind

  save_results_file(dataset_name, single_class_ind, scores=scores,
                    labels=labels, experiment_name='trans-transformations')
  save_results_file(dataset_name, single_class_ind, scores=scores_simple,
                    labels=labels,
                    experiment_name='trans-transformations-simple')
  save_results_file(dataset_name, single_class_ind, scores=scores_entropy,
                    labels=labels,
                    experiment_name='trans-transformations-entropy')
  save_results_file(dataset_name, single_class_ind, scores=scores_xH,
                    labels=labels, experiment_name='trans-transformations-xH')

  mdl_weights_name = '{}_trans-transformations_{}_{}_weights.h5'.format(
      dataset_name,
      get_class_name_from_index(single_class_ind, dataset_name),
      datetime.datetime.now().strftime('%Y-%m-%d-%H%M'))
  mdl_weights_path = os.path.join(RESULTS_DIR, dataset_name, mdl_weights_name)
  mdl.save_weights(mdl_weights_path)


def _kernel_transformations_experiment(dataset_load_fn, dataset_name,
    single_class_ind, gpu_q):
  # gpu_to_use = gpu_q.get()
  # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

  (x_train, y_train), (x_test, y_test) = dataset_load_fn()

  if dataset_name in ['cats-vs-dogs']:
    transformer = None
  else:
    transformer = KernelTransformer(translation_x=8, translation_y=8,
                                    rotations=0,
                                    flips=0, gauss=1, log=1)
    n, k = (10, 4)
  mdl = create_wide_residual_network(x_train.shape[1:],
                                     transformer.n_transforms, n, k)
  mdl.compile('adam',
              'categorical_crossentropy',
              ['acc'])

  # get inliers of specific class
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  # [0_i, ..., (N_transforms-1)_i, ..., ..., 0_N_samples, ...,
  # (N_transforms-1)_N_samples] shape: (N_transforms*N_samples,)
  transformations_inds = np.tile(np.arange(transformer.n_transforms),
                                 len(x_train_task))
  x_train_task_transformed = transformer.transform_batch(
      np.repeat(x_train_task, transformer.n_transforms, axis=0),
      transformations_inds)
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
  scores_entropy = get_entropy(matrix_evals)
  scores_xH = get_xH(transformer, matrix_evals)
  labels = y_test.flatten() == single_class_ind

  save_results_file(dataset_name, single_class_ind, scores=scores,
                    labels=labels, experiment_name='kernel-transformations')
  save_results_file(dataset_name, single_class_ind, scores=scores_simple,
                    labels=labels,
                    experiment_name='kernel-transformations-simple')
  save_results_file(dataset_name, single_class_ind, scores=scores_entropy,
                    labels=labels,
                    experiment_name='kernel-transformations-entropy')
  save_results_file(dataset_name, single_class_ind, scores=scores_xH,
                    labels=labels, experiment_name='kernel-transformations-xH')

  mdl_weights_name = '{}_kernel-transformations_{}_{}_weights.h5'.format(
      dataset_name,
      get_class_name_from_index(single_class_ind, dataset_name),
      datetime.datetime.now().strftime('%Y-%m-%d-%H%M'))
  mdl_weights_path = os.path.join(RESULTS_DIR, dataset_name, mdl_weights_name)
  mdl.save_weights(mdl_weights_path)


def _kernal_plus_transformations_experiment(dataset_load_fn, dataset_name,
    single_class_ind, gpu_q):
  # gpu_to_use = gpu_q.get()
  # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

  (x_train, y_train), (x_test, y_test) = dataset_load_fn()

  if dataset_name in ['cats-vs-dogs']:
    transformer = None
  else:
    transformer = PlusKernelTransformer(translation_x=8, translation_y=8,
                                        rotations=1,
                                        flips=1, gauss=1, log=1)
    n, k = (10, 4)
  mdl = create_wide_residual_network(x_train.shape[1:],
                                     transformer.n_transforms, n, k)
  mdl.compile('adam',
              'categorical_crossentropy',
              ['acc'])

  # get inliers of specific class
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  # [0_i, ..., (N_transforms-1)_i, ..., ..., 0_N_samples, ...,
  # (N_transforms-1)_N_samples] shape: (N_transforms*N_samples,)
  transformations_inds = np.tile(np.arange(transformer.n_transforms),
                                 len(x_train_task))
  x_train_task_transformed = transformer.transform_batch(
    np.repeat(x_train_task, transformer.n_transforms, axis=0),
    transformations_inds)
  batch_size = 128

  mdl.fit(x_train=x_train_task_transformed, y=to_categorical(transformations_inds),
          batch_size=batch_size, epochs=2
          # int(np.ceil(200/transformer.n_transforms))
          )

  scores = np.zeros((len(x_test),))
  matrix_evals = np.zeros(
      (len(x_test), transformer.n_transforms, transformer.n_transforms))
  observed_data = x_train_task
  for t_ind in range(transformer.n_transforms):
    observed_dirichlet = mdl.predict(
      transformer.transform_batch(observed_data, [t_ind] * len(observed_data)),
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

  save_results_file(dataset_name, single_class_ind, scores=scores,
                    labels=labels,
                    experiment_name='kernel-plus-transformations')
  save_results_file(dataset_name, single_class_ind, scores=scores_simple,
                    labels=labels,
                    experiment_name='kernel-plus-transformations-simple')
  save_results_file(dataset_name, single_class_ind, scores=scores_entropy,
                    labels=labels,
                    experiment_name='kernel-plus-transformations-entropy')
  save_results_file(dataset_name, single_class_ind, scores=scores_xH,
                    labels=labels,
                    experiment_name='kernel-plus-transformations-xH')

  mdl_weights_name = '{}_kernel-plus-transformations_{}_{}_weights.h5'.format(
    dataset_name,
    get_class_name_from_index(single_class_ind, dataset_name),
    datetime.datetime.now().strftime('%Y-%m-%d-%H%M'))
  mdl_weights_path = os.path.join(RESULTS_DIR, dataset_name, mdl_weights_name)
  mdl.save_weights(mdl_weights_path)


def _mo_gaal_experiment(dataset_load_fn, dataset_name, single_class_ind):
  (x_train, y_train), (x_test, y_test) = dataset_load_fn()

  x_train = x_train.reshape((len(x_train), -1))
  x_test = x_test.reshape((len(x_test), -1))

  x_train_task = x_train[y_train.flatten() == single_class_ind]

  best_mo_gaal = MO_GAAL().fit(x_train_task)
  scores = best_mo_gaal.decision_function(x_test)
  labels = y_test.flatten() == single_class_ind

  res_file_name = '{}_mo-gaal_{}_{}.npz'.format(dataset_name,
                                                get_class_name_from_index(
                                                    single_class_ind,
                                                    dataset_name),
                                                datetime.datetime.now().strftime(
                                                    '%Y-%m-%d-%H%M'))
  res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
  save_roc_pr_curve_data(scores, labels, res_file_path)


def _train_if_and_score(params, xtrain, test_labels, xtest):
  return roc_auc_score(test_labels,
                       IsolationForest(**params).fit(xtrain).decision_function(
                           xtest))


def _if_experiment(dataset_load_fn, dataset_name, single_class_ind):
  (x_train, y_train), (x_test, y_test) = dataset_load_fn()

  x_train = x_train.reshape((len(x_train), -1))
  x_test = x_test.reshape((len(x_test), -1))

  x_train_task = x_train[y_train.flatten() == single_class_ind]
  # if dataset_name in LARGE_DATASET_NAMES:  # OC-SVM is quadratic on the number of examples, so subsample training set
  #     subsample_inds = np.random.choice(len(x_train_task), 2500, replace=False)
  #     x_train_task_tmp = x_train_task[subsample_inds]

  # ToDO: make gridsearch just one
  pg = ParameterGrid({'n_estimators': np.linspace(100, 800, num=8).astype(int),
                      'contamination': [0.1, 0.2, 0.3, 0.4, 0.5],
                      'behaviour': ['new'],
                      'n_jobs': [-1]})

  # results = Parallel(n_jobs=PARALLEL_N_JOBS)(
  #     delayed(_train_if_and_score)(d, x_train_task, y_test.flatten() == single_class_ind, x_test)
  #     for d in pg)
  results = []
  for d in pg:
    results.append(
        _train_if_and_score(d, x_train_task,
                            y_test.flatten() == single_class_ind, x_test)
    )

  # print(list(zip(pg, results)))
  best_params, best_auc_score = max(zip(pg, results), key=lambda t: t[-1])
  # print(best_params, ' ', best_auc_score)
  best_if = IsolationForest(**best_params).fit(x_train_task)
  scores = best_if.decision_function(x_test)
  labels = y_test.flatten() == single_class_ind

  res_file_name = '{}_if_{}_{}.npz'.format(dataset_name,
                                           get_class_name_from_index(
                                               single_class_ind, dataset_name),
                                           datetime.datetime.now().strftime(
                                               '%Y-%m-%d-%H%M'))
  res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
  save_roc_pr_curve_data(scores, labels, res_file_path)


def _train_ocsvm_and_score(params, xtrain, test_labels, xtest):
  return roc_auc_score(test_labels,
                       OneClassSVM(**params).fit(xtrain).decision_function(
                           xtest))


def _raw_ocsvm_experiment(dataset_load_fn, dataset_name, single_class_ind):
  (x_train, y_train), (x_test, y_test) = dataset_load_fn()

  x_train = x_train.reshape((len(x_train), -1))
  x_test = x_test.reshape((len(x_test), -1))

  x_train_task = x_train[y_train.flatten() == single_class_ind]
  #if dataset_name in LARGE_DATASET_NAMES:  # OC-SVM is quadratic on the number of examples, so subsample training set
  subsample_inds = np.random.choice(len(x_train_task), 2500, replace=False)
  x_train_task_tmp = x_train_task[subsample_inds]

  # ToDO: make gridsearch just one
  pg = ParameterGrid({'nu': np.linspace(0.1, 0.9, num=9),
                      'gamma': np.logspace(-7, 2, num=10, base=2)})

  results = Parallel(n_jobs=PARALLEL_N_JOBS)(
      delayed(_train_ocsvm_and_score)(d, x_train_task_tmp,
                                      y_test.flatten() == single_class_ind,
                                      x_test)
      for d in pg)

  best_params, best_auc_score = max(zip(pg, results), key=lambda t: t[-1])
  print(best_params)
  best_ocsvm = OneClassSVM(**best_params).fit(x_train_task)
  scores = best_ocsvm.decision_function(x_test)
  labels = y_test.flatten() == single_class_ind

  res_file_name = '{}_raw-oc-svm_{}_{}.npz'.format(dataset_name,
                                                   get_class_name_from_index(
                                                       single_class_ind,
                                                       dataset_name),
                                                   datetime.datetime.now().strftime(
                                                       '%Y-%m-%d-%H%M'))
  res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
  save_roc_pr_curve_data(scores, labels, res_file_path)


def _cae_ocsvm_experiment(dataset_load_fn, dataset_name, single_class_ind,
    gpu_q):
  # gpu_to_use = gpu_q.get()
  # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

  (x_train, y_train), (x_test, y_test) = dataset_load_fn()

  print('data_shape', x_train.shape)

  n_channels = x_train.shape[get_channels_axis()]
  input_side = x_train.shape[2]  # channel side will always be at shape[2]
  enc = conv_encoder(input_side, n_channels)
  dec = conv_decoder(input_side, n_channels)
  # print(input_side)
  # print(dec.summary())
  x_in = Input(shape=x_train.shape[1:])
  x_rec = dec(enc(x_in))
  cae = Model(x_in, x_rec)
  cae.compile('adam', 'mse')

  x_train_task = x_train[y_train.flatten() == single_class_ind]
  x_test_task = x_test[
    y_test.flatten() == single_class_ind]  # This is just for visual monitoring
  cae.fit(x_train=x_train_task, y=x_train_task, batch_size=128, epochs=200,
          validation_data=(x_test_task, x_test_task))

  x_train_task_rep = enc.predict(x_train_task, batch_size=128)
  #if dataset_name in LARGE_DATASET_NAMES:  # OC-SVM is quadratic on the number of examples, so subsample training set
  subsample_inds = np.random.choice(len(x_train_task_rep), 2500,
                                      replace=False)
  x_train_task_rep_temp = x_train_task_rep[subsample_inds]

  x_test_rep = enc.predict(x_test, batch_size=128)
  pg = ParameterGrid({'nu': np.linspace(0.1, 0.9, num=9),
                      'gamma': np.logspace(-7, 2, num=10, base=2)})

  results = Parallel(n_jobs=PARALLEL_N_JOBS)(
      delayed(_train_ocsvm_and_score)(d, x_train_task_rep_temp,
                                      y_test.flatten() == single_class_ind,
                                      x_test_rep)
      for d in pg)

  best_params, best_auc_score = max(zip(pg, results), key=lambda t: t[-1])
  print(best_params)
  best_ocsvm = OneClassSVM(**best_params).fit(x_train_task_rep)
  scores = best_ocsvm.decision_function(x_test_rep)
  labels = y_test.flatten() == single_class_ind

  res_file_name = '{}_cae-oc-svm_{}_{}.npz'.format(dataset_name,
                                                   get_class_name_from_index(
                                                       single_class_ind,
                                                       dataset_name),
                                                   datetime.datetime.now().strftime(
                                                       '%Y-%m-%d-%H%M'))
  res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
  save_roc_pr_curve_data(scores, labels, res_file_path)

  # gpu_q.put(gpu_to_use)


def _dsebm_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
  # gpu_to_use = gpu_q.get()
  # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

  (x_train, y_train), (x_test, y_test) = dataset_load_fn()

  n_channels = x_train.shape[get_channels_axis()]
  input_side = x_train.shape[2]  # image side will always be at shape[2]
  encoder_mdl = conv_encoder(input_side, n_channels,
                             representation_activation='relu')
  energy_mdl = dsebm.create_energy_model(encoder_mdl)
  reconstruction_mdl = dsebm.create_reconstruction_model(energy_mdl)

  # optimization parameters
  batch_size = 128
  epochs = 200
  reconstruction_mdl.compile('adam', 'mse')
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  x_test_task = x_test[
    y_test.flatten() == single_class_ind]  # This is just for visual monitoring
  reconstruction_mdl.fit(x_train=x_train_task, y=x_train_task,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(x_test_task, x_test_task))

  scores = -energy_mdl.predict(x_test, batch_size)
  labels = y_test.flatten() == single_class_ind
  res_file_name = '{}_dsebm_{}_{}.npz'.format(dataset_name,
                                              get_class_name_from_index(
                                                  single_class_ind,
                                                  dataset_name),
                                              datetime.datetime.now().strftime(
                                                  '%Y-%m-%d-%H%M'))
  res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
  save_roc_pr_curve_data(scores, labels, res_file_path)

  # gpu_q.put(gpu_to_use)


def _dagmm_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
  # TODO: check cpu usage
  # gpu_to_use = gpu_q.get()
  # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

  (x_train, y_train), (x_test, y_test) = dataset_load_fn()

  n_channels = x_train.shape[get_channels_axis()]
  input_side = x_train.shape[2]  # image side will always be at shape[2]
  enc = conv_encoder(input_side, n_channels, representation_dim=5,
                     representation_activation='linear')
  dec = conv_decoder(input_side, n_channels=n_channels,
                     representation_dim=enc.output_shape[-1])
  n_components = 3
  estimation = Sequential(
      [Dense(64, activation='tanh', input_dim=enc.output_shape[-1] + 2),
       Dropout(0.5),
       Dense(10, activation='tanh'), Dropout(0.5),
       Dense(n_components, activation='softmax')]
  )

  batch_size = 256
  epochs = 200
  lambda_diag = 0.0005
  lambda_energy = 0.01
  dagmm_mdl = dagmm.create_dagmm_model(enc, dec, estimation, lambda_diag)
  dagmm_mdl.compile('adam',
                    ['mse', lambda y_true, y_pred: lambda_energy * y_pred])

  x_train_task = x_train[y_train.flatten() == single_class_ind]
  x_test_task = x_test[
    y_test.flatten() == single_class_ind]  # This is just for visual monitoring
  dagmm_mdl.fit(x_train=x_train_task,
                y=[x_train_task, np.zeros((len(x_train_task), 1))],
                # second y is dummy
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(
                  x_test_task, [x_test_task, np.zeros((len(x_test_task), 1))]),
                # verbose=0
                )

  energy_mdl = Model(dagmm_mdl.input, dagmm_mdl.output[-1])

  scores = -energy_mdl.predict(x_test, batch_size)
  scores = scores.flatten()
  if not np.all(np.isfinite(scores)):
    min_finite = np.min(scores[np.isfinite(scores)])
    scores[~np.isfinite(scores)] = min_finite - 1
  labels = y_test.flatten() == single_class_ind
  res_file_name = '{}_dagmm_{}_{}.npz'.format(dataset_name,
                                              get_class_name_from_index(
                                                  single_class_ind,
                                                  dataset_name),
                                              datetime.datetime.now().strftime(
                                                  '%Y-%m-%d-%H%M'))
  res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
  save_roc_pr_curve_data(scores, labels, res_file_path)

  # gpu_q.put(gpu_to_use)


def _adgan_experiment(dataset_load_fn, dataset_name, single_class_ind, gpu_q):
  # gpu_to_use = gpu_q.get()
  # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use

  (x_train, y_train), (x_test, y_test) = dataset_load_fn()
  if len(x_test) > 5000:
    # subsample x_test due to runtime complexity
    chosen_inds = np.random.choice(len(x_test), 5000, replace=False)
    x_test = x_test[chosen_inds]
    y_test = y_test[chosen_inds]

  n_channels = x_train.shape[get_channels_axis()]
  input_side = x_train.shape[2]  # image side will always be at shape[2]
  critic = conv_encoder(input_side, n_channels, representation_dim=1,
                        representation_activation='linear')
  noise_size = 256
  generator = conv_decoder(input_side, n_channels=n_channels,
                           representation_dim=noise_size)

  def prior_gen(b_size):
    return np.random.normal(size=(b_size, noise_size))

  batch_size = 128
  epochs = 100

  x_train_task = x_train[y_train.flatten() == single_class_ind]

  def data_gen(b_size):
    chosen_inds = np.random.choice(len(x_train_task), b_size, replace=False)
    return x_train_task[chosen_inds]

  adgan.train_wgan_with_grad_penalty(prior_gen, generator, data_gen, critic,
                                     batch_size, epochs, grad_pen_coef=20)

  scores = adgan.scores_from_adgan_generator(x_test, prior_gen, generator)
  labels = y_test.flatten() == single_class_ind
  res_file_name = '{}_adgan_{}_{}.npz'.format(dataset_name,
                                              get_class_name_from_index(
                                                  single_class_ind,
                                                  dataset_name),
                                              datetime.datetime.now().strftime(
                                                  '%Y-%m-%d-%H%M'))
  res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
  save_roc_pr_curve_data(scores, labels, res_file_path)

  # gpu_q.put(gpu_to_use)


# TODO: check real parallelism of tasks
# ToDo: research how to perform multi gpu training
def run_experiments(load_dataset_fn, dataset_name, q, class_idx, n_runs):
  check_path(os.path.join(RESULTS_DIR, dataset_name))
  # # Kernel-plus-Transformations
  # for _ in range(n_runs):
  #     processes = [Process(target=_kernal_plus_transformations_experiment,
  #                          args=(
  #                          load_dataset_fn, dataset_name, class_idx, q))]
  #     for p in processes:
  #         p.start()
  #     for p in processes:
  #         p.join()
  #
  # # MO_GAAL
  # for _ in range(n_runs):
  #   _mo_gaal_experiment(load_dataset_fn, dataset_name, class_idx)
  #
  # # IF
  # for _ in range(n_runs):
  #   _if_experiment(load_dataset_fn, dataset_name, class_idx)
  #
  ## CAE OC-SVM
  # for _ in range(n_runs):
  #   processes = [Process(target=_cae_ocsvm_experiment,
  #                        args=(load_dataset_fn, dataset_name, class_idx, q))]
  #   for p in processes:
  #     p.start()
  #     p.join()

  # Raw OC-SVM
  for _ in range(n_runs):
    _raw_ocsvm_experiment(load_dataset_fn, dataset_name, class_idx)

  # # Transformations
  # for _ in range(n_runs):
  #    processes = [Process(target=_transformations_experiment,
  #                         args=(load_dataset_fn, dataset_name, class_idx, q))]
  #    for p in processes:
  #        p.start()
  #    for p in processes:
  #        p.join()
  # #
  # # Trans-Transformations
  # for _ in range(n_runs):
  #    processes = [Process(target=_trans_transformations_experiment,
  #                         args=(load_dataset_fn, dataset_name, class_idx, q))]
  #    for p in processes:
  #        p.start()
  #    for p in processes:
  #        p.join()
  #
  ## Kernel-Transformations
  # for _ in range(n_runs):
  #    processes = [Process(target=_kernel_transformations_experiment,
  #                         args=(
  #                         load_dataset_fn, dataset_name, class_idx, q))]
  #    for p in processes:
  #        p.start()
  #    for p in processes:
  #        p.join()
  #
  # DSEBM
  for _ in range(n_runs):
    processes = [Process(target=_dsebm_experiment,
                         args=(load_dataset_fn, dataset_name, class_idx, q))]
    for p in processes:
      p.start()
    for p in processes:
      p.join()

  # # DAGMM
  # for _ in range(n_runs):
  #     processes = [Process(target=_dagmm_experiment,
  #                              args=(load_dataset_fn, dataset_name, class_idx, q))]
  #     for p in processes:
  #         p.start()
  #     for p in processes:
  #         p.join()

  # ADGAN
  for _ in range(n_runs):
    processes = [Process(target=_adgan_experiment,
                         args=(load_dataset_fn, dataset_name, class_idx, q))]
    for p in processes:
      p.start()
    for p in processes:
      p.join()


def create_auc_table(metric='roc_auc'):
  file_path = glob(os.path.join(RESULTS_DIR, '*', '*.npz'))
  results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  methods = set()
  for p in file_path:
    _, f_name = os.path.split(p)
    dataset_name, method, single_class_name = f_name.split(sep='_')[:3]
    methods.add(method)
    npz = np.load(p)
    roc_auc = npz[metric]
    results[dataset_name][single_class_name][method].append(roc_auc)

  for ds_name in results:
    for sc_name in results[ds_name]:
      for method_name in results[ds_name][sc_name]:
        roc_aucs = results[ds_name][sc_name][method_name]
        print(method_name, ' ', roc_aucs)
        results[ds_name][sc_name][method_name] = [np.mean(roc_aucs),
                                                  0 if len(
                                                      roc_aucs) == 1 else scipy.stats.sem(
                                                      np.array(roc_aucs))
                                                  ]

  with open(os.path.join(RESULTS_DIR, 'results-{}.csv'.format(metric)),
            'w') as csvfile:
    fieldnames = ['dataset', 'single class name'] + sorted(list(methods))
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ds_name in sorted(results.keys()):
      for sc_name in sorted(results[ds_name].keys()):
        row_dict = {'dataset': ds_name, 'single class name': sc_name}
        row_dict.update({method_name: '{:.5f} ({:.5f})'.format(
            *results[ds_name][sc_name][method_name])
          for method_name in results[ds_name][sc_name]})
        writer.writerow(row_dict)


if __name__ == '__main__':
  freeze_support()
  N_GPUS = 1
  man = Manager()
  q = man.Queue(N_GPUS)
  for g in range(N_GPUS):
    q.put(str(g))

  # data_Set, dataset_name, class_idx_to_run_experiments_on-inlier_class, n_runs
  experiments_list = [
    (base_line_loaders.load_ztf_small, 'ztf-small-real-bog', 1, 10),
    (base_line_loaders.load_hits4c_outlier_loader, 'hits-4-c-od', 1, 10)
  ]

  start_time = time.time()
  for data_load_fn, dataset_name, class_idx, run_i in experiments_list:
    run_experiments(data_load_fn, dataset_name, q, class_idx, run_i)
  # create_auc_table()
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time elapsed to train everything: " + time_usage)
