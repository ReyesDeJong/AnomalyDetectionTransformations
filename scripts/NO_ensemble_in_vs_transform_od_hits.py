import os
import sys

"""
trying to detect outlier by training classifierS that distinguish between inliers and an specific transform.
This is illposed because classifier never sees another type of transforms
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
from keras.utils import to_categorical
from modules.data_loaders.base_line_loaders import load_hits

from transformations import TransTransformer
from models.simple_network import create_simple_network
import time
import datetime
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from tqdm import tqdm
from scripts.detached_transformer_od_hits import plot_histogram_disc_loss_acc_thr, \
  dirichlet_normality_score, fixed_point_dirichlet_mle, calc_approx_alpha_sum
import matplotlib.pyplot as plt

if __name__ == "__main__":
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
  sess = tf.Session(config=config)
  set_session(sess)

  single_class_ind = 1

  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_hits(n_samples_by_class=16000,
                                                   test_size=0.25,
                                                   val_size=0.125, return_val=True)
  print(x_train.shape)
  print(x_val.shape)
  print(x_test.shape)

  transformer = TransTransformer(8, 8)
  # n, k = (10, 4)
  #
  # mdl = create_wide_residual_network(input_shape=x_train.shape[1:],
  #                                    num_classes=transformer.n_transforms,
  #                                    depth=n, widen_factor=k)

  mdl = create_simple_network(input_shape=x_train.shape[1:],
                              num_classes=2, dropout_rate=0.5)
  mdl.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])

  print(mdl.summary())



  # get inliers of specific class
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  print(x_train_task.shape)

  x_val_task = x_val[y_val.flatten() == single_class_ind]
  print(x_val_task.shape)
  # [0_i, ..., (N_transforms-1)_i, ..., ..., 0_N_samples, ...,
  # (N_transforms-1)_N_samples] shape: (N_transforms*N_samples,)
  transformations_inds_train = np.tile(np.arange(transformer.n_transforms),
                                 len(x_train_task))
  transformations_inds_val = np.tile(np.arange(transformer.n_transforms),
                                 len(x_val_task))
  print(len(transformations_inds_train))
  print(len(transformations_inds_val))
  #
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
  #
  transformation_to_train = 3
  # while transformation_to_train!=0:
  #   transformation_to_train = np.random.choice(transformations_inds_train, 1)

  normal_indxs_train = np.where(transformations_inds_train == 0)[0]
  transform_indxs_train = np.where(transformations_inds_train == transformation_to_train)[0]
  normal_indxs_val = np.where(transformations_inds_val == 0)[0]
  transform_indxs_val = \
  np.where(transformations_inds_val == transformation_to_train)[0]

  train_x_binary = np.concatenate([x_train_task_transformed[normal_indxs_train], x_train_task_transformed[transform_indxs_train]])
  train_y_binary = np.concatenate([np.zeros_like(normal_indxs_train), np.ones_like(normal_indxs_train)])
  val_x_binary = np.concatenate([x_val_task_transformed[normal_indxs_val],
                                   x_val_task_transformed[
                                     transform_indxs_val]])
  val_y_binary = np.concatenate(
      [np.zeros_like(normal_indxs_val), np.ones_like(normal_indxs_val)])



  plt.imshow(train_x_binary[0, ..., 0])
  plt.show()
  plt.imshow(train_x_binary[10000, ..., 0])
  plt.show()


  start_time = time.time()
  mdl.fit(x=train_x_binary, y=to_categorical(train_y_binary), validation_data=(val_x_binary, to_categorical(val_y_binary)),
          batch_size=batch_size,
          epochs=1,  # int(np.ceil(200 / transformer.n_transforms))
          )
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time to train model: " + time_usage)

  # scores = np.zeros((len(x_test),))
  # observed_data = x_train_task
  #
  # # # testing inside for
  # # t_ind = np.random.randint(transformer.n_transforms)
  # # observed_dirichlet = mdl.predict(
  # #     transformer.transform_batch(observed_data, [t_ind] * len(observed_data)),
  # #     batch_size=1024)
  # # predicted_labels = np.argmax(observed_dirichlet, axis=-1)
  # # print('index to predict: ', t_ind, '\nPredicted counts: ',
  # #       np.unique(predicted_labels, return_counts=True))
  # # log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)
  # # print('log_p_hat_train.shape: ', log_p_hat_train.shape)
  # # alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
  # # print('alpha_sum_approx.shape: ', alpha_sum_approx.shape)
  # # alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx
  # # print('alpha_0.shape: ', alpha_0.shape)
  # # mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
  # # print('mle_alpha_t.shape: ', mle_alpha_t.shape)
  # # x_test_p = mdl.predict(
  # #     transformer.transform_batch(x_test, [t_ind] * len(x_test)),
  # #     batch_size=1024)
  # # predicted_test_labels = np.argmax(x_test_p, axis=-1)
  # # print('index to predict: ', t_ind, '\nPredicted test counts: ',
  # #       np.unique(predicted_test_labels, return_counts=True))
  # #
  # # score_for_specific_transform = dirichlet_normality_score(mle_alpha_t,
  # #                                                          x_test_p)
  # # print('score_for_specific_transform.shape: ',
  # #       score_for_specific_transform.shape)
  #
  # # Dirichlet transforms
  # for t_ind in tqdm(range(transformer.n_transforms)):
  #   # predictions for a single transformation
  #   observed_dirichlet = mdl.predict(
  #       transformer.transform_batch(observed_data,
  #                                   [t_ind] * len(observed_data)),
  #       batch_size=1024)
  #   log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)
  #
  #   alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
  #   alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx
  #
  #   mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
  #
  #   x_test_p = mdl.predict(
  #       transformer.transform_batch(x_test, [t_ind] * len(x_test)),
  #       batch_size=1024)
  #   scores += dirichlet_normality_score(mle_alpha_t, x_test_p)
  #
  # scores /= transformer.n_transforms
  # labels = y_test.flatten() == single_class_ind
  #
  # plot_histogram_disc_loss_acc_thr(scores[labels], scores[~labels],
  #                                  x_label_name='TransTransformations_Dscores_hits')
  #
  # # Dirichlet transforms with arcsin
  # neg_scores = -scores
  # norm_scores = neg_scores - np.min(neg_scores)
  # norm_scores = norm_scores / np.max(norm_scores)
  # arcsinh_scores = np.arcsinh(norm_scores * 10000)
  # inlier_arcsinh_score = arcsinh_scores[labels]
  # outlier_arcsinh_score = arcsinh_scores[~labels]
  # plot_histogram_disc_loss_acc_thr(inlier_arcsinh_score, outlier_arcsinh_score,
  #                                  '../results',
  #                                  'TransTransformations_arcsinh*10000_Dscores')
  #
  # # Transforms without dirichlet
  # plain_scores = np.zeros((len(x_test),))
  # for t_ind in tqdm(range(transformer.n_transforms)):
  #   # predictions for a single transformation
  #   x_test_p = mdl.predict(
  #       transformer.transform_batch(x_test, [t_ind] * len(x_test)),
  #       batch_size=1024)
  #   plain_scores += x_test_p[:, t_ind]
  #
  # plain_scores /= transformer.n_transforms
  # labels = y_test.flatten() == single_class_ind
  #
  # plot_histogram_disc_loss_acc_thr(plain_scores[labels], plain_scores[~labels],
  #                                  x_label_name='TransTransformations_scores_hits')
  #
  # # Transforms without dirichlet arcsinh
  # plain_neg_scores = -plain_scores
  # plain_norm_scores = plain_neg_scores - np.min(plain_neg_scores)
  # plain_norm_scores = plain_norm_scores / plain_norm_scores.max()
  # plain_arcsinh_scores = np.arcsinh(plain_norm_scores * 10000)
  #
  # plot_histogram_disc_loss_acc_thr(plain_arcsinh_scores[labels],
  #                                  plain_arcsinh_scores[~labels],
  #                                  x_label_name='TransTransformations_arcsinh*10000_scores_hits')
