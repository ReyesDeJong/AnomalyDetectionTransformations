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
from modules.data_loaders.base_line_loaders import load_hits

from transformations import KernelTransformer
from models.wide_residual_network import create_wide_residual_network
import time
import datetime
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from tqdm import tqdm
from scripts.detached_transformer_od_hits import \
  plot_histogram_disc_loss_acc_thr, \
  dirichlet_normality_score, fixed_point_dirichlet_mle, calc_approx_alpha_sum
from scripts.ensemble_transform_vs_all_od_hits import get_entropy
import torch
import torch.nn as nn

if __name__ == "__main__":
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
  sess = tf.Session(config=config)
  set_session(sess)

  single_class_ind = 1

  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_hits(
      n_samples_by_class=10000,
      test_size=0.20,
      val_size=0.10, return_val=True)
  print(x_train.shape)
  print(x_val.shape)
  print(x_test.shape)

  transformer = KernelTransformer(translation_x=0, translation_y=0, rotations=0,
                                  flips=0, gauss=1, log=1)
  n, k = (10, 4)

  mdl = create_wide_residual_network(input_shape=x_train.shape[1:],
                                     num_classes=transformer.n_transforms,
                                     depth=n, widen_factor=k)
  mdl.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])

  print(mdl.summary())

  print('n_transforms ', transformer.n_transforms)
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
  mdl.fit(x_train=x_train_task_transformed,
          y=to_categorical(transformations_inds_train),
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
                                   path='../results',
                                   x_label_name='KernelOnlyTransformations_Dscores_hits',
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
  # val
  plain_scores_val = np.zeros((len(x_val_task),))
  for t_ind in tqdm(range(transformer.n_transforms)):
    # predictions for a single transformation
    x_val_p = mdl.predict(
        transformer.transform_batch(x_val_task, [t_ind] * len(x_val_task)),
        batch_size=1024)
    plain_scores_val += x_val_p[:, t_ind]

  plain_scores_val /= transformer.n_transforms

  labels = y_test.flatten() == single_class_ind

  plot_histogram_disc_loss_acc_thr(plain_scores_test[labels],
                                   plain_scores_test[~labels],
                                   path='../results',
                                   x_label_name='KernelOnlyTransformations_scores_hits',
                                   val_inliers_score=plain_scores_val)