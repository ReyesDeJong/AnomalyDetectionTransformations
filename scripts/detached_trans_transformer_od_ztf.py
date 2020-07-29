import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

"""
Original model, but only with translation transformations (9 transformations), original Resnet is used
"""

import numpy as np
from keras.utils import to_categorical
from modules.data_loaders.base_line_loaders import load_ztf_real_bog

from transformations import TransTransformer
from models.wide_residual_network import create_wide_residual_network
import time
import datetime
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from tqdm import tqdm
from scripts.detached_transformer_od_hits import \
  plot_histogram_disc_loss_acc_thr, \
  dirichlet_normality_score, fixed_point_dirichlet_mle, calc_approx_alpha_sum
from scripts.ensemble_transform_vs_all_od_hits import get_entropy, \
  plot_matrix_score
import torch
import torch.nn as nn
from modules.utils import check_paths

EXPERIMENT_NAME = 'ZTF_v1_TransTransformations'

if __name__ == "__main__":
  results_folder = os.path.join(PROJECT_PATH, 'results', EXPERIMENT_NAME)
  check_paths(results_folder)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
  sess = tf.Session(config=config)
  set_session(sess)

  single_class_ind = 1

  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_ztf_real_bog(return_val=True)#, data_file_name='ztf_v1_bogus_added.pkl')
  print(x_train.shape)
  print(x_val.shape)
  print(x_test.shape)
  # (x_train, y_train) = (x_train[:1000], y_train[:1000])
  # (x_val, y_val) = (x_val[:1000], y_val[:1000])


  transformer = TransTransformer(8, 8)
  n, k = (10, 4)

  mdl = create_wide_residual_network(input_shape=x_train.shape[1:],
                                     num_classes=transformer.n_transforms,
                                     depth=n, widen_factor=k)
  mdl.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])

  print(mdl.summary())

  # get inliers of specific class
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  print(x_train_task.shape)

  x_val_task = x_val[y_val.flatten() == single_class_ind]
  print(x_val_task.shape)

  transformations_inds_train = np.tile(np.arange(transformer.n_transforms),
                                       len(x_train_task))
  transformations_inds_val = np.tile(np.arange(transformer.n_transforms),
                                     len(x_val_task))
  print(len(transformations_inds_train))
  print(len(transformations_inds_val))

  # transform data
  start_time = time.time()
  x_train_task_transformed = transformer.transform_batch(
      np.repeat(x_train_task, transformer.n_transforms, axis=0),
      transformations_inds_train)
  x_val_task_transformed = transformer.transform_batch(
      np.repeat(x_val_task, transformer.n_transforms, axis=0),
      transformations_inds_val)
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time to perform transforms: " + time_usage)
  print(x_train_task_transformed.shape)
  print(x_val_task_transformed.shape)
  batch_size = 128

  start_time = time.time()
  mdl.fit(x_train=x_train_task_transformed, y=to_categorical(transformations_inds_train),
          batch_size=batch_size,
          epochs=2,  # int(np.ceil(200 / transformer.n_transforms))
          )
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time to train model: " + time_usage)

  test_scores = np.zeros((len(x_test),))
  val_scores_in = np.zeros((len(x_val_task),))
  observed_data = x_train_task

  # Dirichlet transforms
  for t_ind in tqdm(range(transformer.n_transforms)):
    # predictions for a single transformation
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
    test_scores += dirichlet_normality_score(mle_alpha_t, x_test_p)

  test_scores /= transformer.n_transforms

  # val
  # Dirichlet transforms
  for t_ind in tqdm(range(transformer.n_transforms)):
    # predictions for a single transformation
    observed_dirichlet = mdl.predict(
        transformer.transform_batch(observed_data,
                                    [t_ind] * len(observed_data)),
        batch_size=1024)
    log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

    alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
    alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

    mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)

    x_val_p = mdl.predict(
        transformer.transform_batch(x_val_task, [t_ind] * len(x_val_task)),
        batch_size=1024)
    val_scores_in += dirichlet_normality_score(mle_alpha_t, x_val_p)

  val_scores_in /= transformer.n_transforms

  labels = y_test.flatten() == single_class_ind

  plot_histogram_disc_loss_acc_thr(test_scores[labels], test_scores[~labels],
                                   path=results_folder,
                                   x_label_name='Dscores',
                                   val_inliers_score=val_scores_in)

  # Transforms without dirichlet
  plain_scores_test = np.zeros((len(x_test),))
  for t_ind in tqdm(range(transformer.n_transforms)):
    # predictions for a single transformation
    x_test_p = mdl.predict(
        transformer.transform_batch(x_test, [t_ind] * len(x_test)),
        batch_size=1024)
    plain_scores_test += x_test_p[:, t_ind]

  plain_scores_test /= transformer.n_transforms
  #val
  plain_scores_val = np.zeros((len(x_val_task),))
  for t_ind in tqdm(range(transformer.n_transforms)):
    # predictions for a single transformation
    x_val_p = mdl.predict(
        transformer.transform_batch(x_val_task, [t_ind] * len(x_val_task)),
        batch_size=1024)
    plain_scores_val += x_val_p[:, t_ind]

  plain_scores_val /= transformer.n_transforms

  labels = y_test.flatten() == single_class_ind

  plot_histogram_disc_loss_acc_thr(plain_scores_test[labels], plain_scores_test[~labels],
                                   path=results_folder,
                                   x_label_name='scores',
                                   val_inliers_score=plain_scores_val)


  ## matrices
  # transform test
  transformations_inds_test = np.tile(np.arange(transformer.n_transforms),
                                      len(x_test))
  start_time = time.time()
  x_test_transformed = transformer.transform_batch(
      np.repeat(x_test, transformer.n_transforms, axis=0),
      transformations_inds_test)
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time to perform transforms: " + time_usage)


  # Get matrix scores
  matrix_scores_test = np.zeros(
      (len(x_test), transformer.n_transforms, transformer.n_transforms))
  for t_ind in tqdm(range(transformer.n_transforms)):
    test_specific_transform_indxs = np.where(
        transformations_inds_test == t_ind)
    x_test_specific_transform = x_test_transformed[
      test_specific_transform_indxs]
    # predictions for a single transformation
    x_test_p = mdl.predict(x_test_specific_transform, batch_size=64)
    matrix_scores_test[:, :, t_ind] += x_test_p

  matrix_scores_test /= transformer.n_transforms
  #val
  matrix_scores_val = np.zeros(
      (len(x_val_task), transformer.n_transforms, transformer.n_transforms))
  for t_ind in tqdm(range(transformer.n_transforms)):
    val_specific_transform_indxs = np.where(
        transformations_inds_val == t_ind)
    x_val_specific_transform = x_val_task_transformed[
      val_specific_transform_indxs]
    # predictions for a single transformation
    x_val_p = mdl.predict(x_val_specific_transform, batch_size=64)
    matrix_scores_val[:, :, t_ind] += x_val_p

  matrix_scores_val /= transformer.n_transforms
  labels = y_test.flatten() == single_class_ind

  # plot_matrix_score(x_test, matrix_scores, labels, plot_inliers=True,
  #                   n_to_plot=5)
  # plot_matrix_score(x_test, matrix_scores, labels, plot_inliers=False,
  #                   n_to_plot=5)

  plot_histogram_disc_loss_acc_thr(
      np.trace(matrix_scores_test[labels], axis1=1, axis2=2),
      np.trace(matrix_scores_test[~labels], axis1=1, axis2=2), path=results_folder,
      x_label_name='matrixTrace',
      val_inliers_score=np.trace(matrix_scores_val))

  entropy_scores_test = get_entropy(matrix_scores_test)
  entropy_scores_val = get_entropy(matrix_scores_val)
  plot_histogram_disc_loss_acc_thr(entropy_scores_test[labels],
                                   entropy_scores_test[~labels],
                                   path=results_folder,
                                   x_label_name='entropy_scores',
                                   val_inliers_score=entropy_scores_val)

  ## Get logits for xentropy
  # Get matrix scores
  matrix_scores_raw_test = np.zeros(
      (len(x_test), transformer.n_transforms, transformer.n_transforms))
  for t_ind in tqdm(range(transformer.n_transforms)):
    test_specific_transform_indxs = np.where(
        transformations_inds_test == t_ind)
    x_test_specific_transform = x_test_transformed[
      test_specific_transform_indxs]
    # predictions for a single transformation
    x_test_p = mdl.predict(x_test_specific_transform, batch_size=64)
    matrix_scores_raw_test[:, :, t_ind] += x_test_p

  matrix_score_compĺement_test = 1 - matrix_scores_raw_test

  matrix_scores_stack_test = np.stack([matrix_score_compĺement_test, matrix_scores_raw_test],
                                      axis=-1)

  xH = nn.NLLLoss(reduction='none')
  gt_matrix = np.stack(
    [np.eye(transformer.n_transforms)] * len(matrix_scores_stack_test))
  gt_torch = torch.LongTensor(gt_matrix)

  matrix_logSoftmax_torch = torch.FloatTensor(
    np.swapaxes(np.swapaxes(matrix_scores_stack_test, 1, -1), -1, -2)).log()
  loss_xH = xH(matrix_logSoftmax_torch, gt_torch)
  batch_xH_test = np.mean(loss_xH.numpy(), axis=(-1, -2))

  #val
  matrix_scores_raw_val = np.zeros(
      (len(x_val_task), transformer.n_transforms, transformer.n_transforms))
  for t_ind in tqdm(range(transformer.n_transforms)):
    val_specific_transform_indxs = np.where(
        transformations_inds_val == t_ind)
    x_val_specific_transform = x_val_task_transformed[
      val_specific_transform_indxs]
    # predictions for a single transformation
    x_val_p = mdl.predict(x_val_specific_transform, batch_size=64)
    matrix_scores_raw_val[:, :, t_ind] += x_val_p

  matrix_score_compĺement_val = 1 - matrix_scores_raw_val

  matrix_scores_stack_val = np.stack([matrix_score_compĺement_val, matrix_scores_raw_val],
                                      axis=-1)

  xH = nn.NLLLoss(reduction='none')
  gt_matrix = np.stack(
    [np.eye(transformer.n_transforms)] * len(matrix_scores_stack_val))
  gt_torch = torch.LongTensor(gt_matrix)

  matrix_logSoftmax_torch = torch.FloatTensor(
    np.swapaxes(np.swapaxes(matrix_scores_stack_val, 1, -1), -1, -2)).log()
  loss_xH = xH(matrix_logSoftmax_torch, gt_torch)
  batch_xH_val = np.mean(loss_xH.numpy(), axis=(-1, -2))

  plot_histogram_disc_loss_acc_thr(batch_xH_test[labels],
                                   batch_xH_test[~labels],
                                   path=results_folder,
                                   x_label_name='xH_scores',
                                   val_inliers_score=batch_xH_val)

  # get worst n traces for inliers and best n traces for outliers

  in_matrix_score = matrix_scores_test[labels]
  out_matrix_score = matrix_scores_test[~labels]
  indx_in = np.argsort(np.trace(in_matrix_score, axis1=1, axis2=2))
  indx_out = np.argsort(np.trace(out_matrix_score, axis1=1, axis2=2))
  x_test_in = x_test[labels]
  x_test_out = x_test[~labels]

  # # best in
  # plot_matrix_score(x_test_in, in_matrix_score, n_to_plot=indx_in[-3:],
  #                   plot_inliers=True)
  # # worst in
  # plot_matrix_score(x_test_in, in_matrix_score, n_to_plot=indx_in[:3],
  #                   plot_inliers=True)
  # # worst outliers out (high trace)
  # plot_matrix_score(x_test_out, out_matrix_score, n_to_plot=indx_out[-3:],
  #                   plot_inliers=False)
  # # best outliers out (low trace)
  # plot_matrix_score(x_test_out, out_matrix_score, n_to_plot=indx_out[:3],
  #                   plot_inliers=False)
