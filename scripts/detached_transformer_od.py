import os, sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
from scipy.special import psi, polygamma
from keras.utils import to_categorical
from modules.data_loaders.base_line_loaders import load_hits

from transformations import Transformer
from models.wide_residual_network import create_wide_residual_network



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

if __name__ == "__main__":
  single_class_ind = 1

  (x_train, y_train), (x_test, y_test) = load_hits(n_samples_by_class=10000, test_size=0.4, val_size=0.09999999)
  print(x_train.shape)
  print(x_test.shape)

  transformer = Transformer(8, 8)
  n, k = (10, 4)

  mdl = create_wide_residual_network(input_shape=x_train.shape[1:], num_classes=transformer.n_transforms, depth=n, widen_factor=k)
  mdl.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

  print(mdl.summary())

  # get inliers of specific class
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  # [0_i, ..., (N_transforms-1)_i, ..., ..., 0_N_samples, ...,
  # (N_transforms-1)_N_samples] shape: (N_transforms*N_samples,)
  transformations_inds = np.tile(np.arange(transformer.n_transforms),
                                 len(x_train_task))
  #
  # x_train_task_transformed = transformer.transform_batch(
  #   np.repeat(x_train_task, transformer.n_transforms, axis=0),
  #   transformations_inds)
  # batch_size = 128
  #
  # mdl.fit(x=x_train_task_transformed, y=to_categorical(transformations_inds),
  #         batch_size=batch_size,
  #         epochs=int(np.ceil(200 / transformer.n_transforms))
  #         )
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
