import os, sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import psi, polygamma
from keras.utils import to_categorical
from modules.data_loaders.base_line_loaders import load_fashion_mnist
from transformations import Transformer
from models.wide_residual_network import create_wide_residual_network
import time
import datetime
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from scripts.detached_transformer_od_hits import calc_approx_alpha_sum, inv_psi, fixed_point_dirichlet_mle, dirichlet_normality_score

if __name__ == "__main__":
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
  sess = tf.Session(config=config)
  set_session(sess)

  single_class_ind = 1

  (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
  print(x_train.shape)
  print(x_test.shape)

  transformer = Transformer(8, 8)
  n, k = (10, 4)

  mdl = create_wide_residual_network(input_shape=x_train.shape[1:], num_classes=transformer.n_transforms, depth=n, widen_factor=k)
  mdl.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

  print(mdl.summary())

  # get inliers of specific class
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  print(x_train_task.shape)
  # [0_i, ..., (N_transforms-1)_i, ..., ..., 0_N_samples, ...,
  # (N_transforms-1)_N_samples] shape: (N_transforms*N_samples,)
  transformations_inds = np.tile(np.arange(transformer.n_transforms),
                                 len(x_train_task))
  print(len(transformations_inds))
  #
  start_time = time.time()
  x_train_task_transformed = transformer.transform_batch(
    np.repeat(x_train_task, transformer.n_transforms, axis=0),
    transformations_inds)
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time to perform transforms: " + time_usage)
  print(x_train_task_transformed.shape)
  batch_size = 128

  start_time = time.time()
  mdl.fit(x=x_train_task_transformed, y=to_categorical(transformations_inds),
          batch_size=batch_size,
          epochs=int(np.ceil(200 / transformer.n_transforms))
          )
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time to train model: " + time_usage)
  #
  # scores = np.zeros((len(x_test),))
  # observed_data = x_train_task
  # for t_ind in range(transformer.n_transforms):
  #     observed_dirichlet = mdl.predict(transformer.transform_batch(observed_data, [t_ind] * len(observed_data)),
  #                                      batch_size=1024)
  #     log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)
  #
  #     alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
  #     alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx
  #
  #     mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
  #
  #     x_test_p = mdl.predict(transformer.transform_batch(x_test, [t_ind] * len(x_test)),
  #                            batch_size=1024)
  #     scores += dirichlet_normality_score(mle_alpha_t, x_test_p)
  #
  # scores /= transformer.n_transforms
  # labels = y_test.flatten() == single_class_ind
  #
