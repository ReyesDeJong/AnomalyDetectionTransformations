import os
import sys

import numpy as np

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from scipy.special import psi, polygamma

#ToDo: see if this can be done by tf

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

def dirichlet_score(predict_x_train, predict_x_eval):
  observed_dirichlet = predict_x_train
  x_eval_p = predict_x_eval

  log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)
  alpha_sum_approx = calc_approx_alpha_sum(
      observed_dirichlet)
  alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx
  mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
  diri_score = dirichlet_normality_score(mle_alpha_t, x_eval_p)
  return diri_score