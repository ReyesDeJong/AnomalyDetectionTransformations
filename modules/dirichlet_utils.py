import os
import sys

import numpy as np

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from scipy.special import psi, polygamma
from modules.utils import normalize_sum1

# ToDo: see if this can be done by tf

def calc_approx_alpha_sum(observations):
  N = len(observations)
  f = np.mean(observations, axis=0)

  return (N * (len(f) - 1) * (-psi(1))) / (
      N * np.sum(f * np.log(f)) - np.sum(
      f * np.sum(np.log(observations), axis=0)))


def inv_psi(y, iters=5):
  # initial estimate
  # if not np.isfinite(y).any():
  #   print('')
  cond = y >= -2.22
  # if not np.isfinite(y).all():
  #   print('problem y')
  # if not np.isfinite(np.exp(y)).all():
  #   print('problem exp')
  # psi(1)= -0.5772156649015329
  # y-psi(1) can give 0 and turns divition to NAN, but this divition is only
  # performed when y < -2.22, so in practice y-psi(1) where y=psi(1) never
  # happens
  x = (1 - cond) * -1 / ((y + 1e-6) - psi(1))
  x[~np.isfinite(x)] = 0
  x = cond * (np.exp(y) + 0.5) + x

  for _ in range(iters):
    x = x - (psi(x) - y) / polygamma(1, x)
  return x


def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
  alpha_new = alpha_old = alpha_init
  for i in range(max_iter):
    # if not np.isfinite(alpha_old).all():
    #   print('problem alpha_old')
    alpha_old_sum=np.sum(alpha_old)
    # if alpha_old_sum%1==0 and alpha_old_sum<0:
    #   alpha_old_sum=+0.5
    alpha_new = inv_psi(psi(alpha_old_sum) + log_p_hat)
    if not np.isfinite(alpha_new).all() or np.sum(alpha_new)<0:
      log_file_path = os.path.join(PROJECT_PATH, 'results', '1a_error.log')
      log_file = open(log_file_path, 'w')
      log_file.write(str(alpha_init)+'\n')
      log_file.write(str(alpha_old)+'\n')
      log_file.write(str(log_p_hat)+'\n')
      log_file.write(str(alpha_old_sum)+'\n')
      log_file.write(str(psi(alpha_old_sum))+'\n')
      log_file.write(str(psi(alpha_old_sum) + log_p_hat)+'\n')
      log_file.close()
      assert np.isfinite(alpha_new).all() or np.sum(alpha_new)<0
    if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9:
      break
    # if not np.isfinite(alpha_new).all():
    #   print('problem alpha_new')
    alpha_old = alpha_new
  return alpha_new


def dirichlet_normality_score(alpha, p):
  return np.sum((alpha - 1) * np.log(p), axis=-1)


def dirichlet_score(predict_x_train, predict_x_eval):
  #TODO: test without this

  # observed_dirichlet = predict_x_train
  # x_eval_p = predict_x_eval
  observed_dirichlet = correct_0_value_predictions(predict_x_train)
  x_eval_p = correct_0_value_predictions(predict_x_eval)

  log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)
  alpha_sum_approx = calc_approx_alpha_sum(
      observed_dirichlet)
  alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx
  if np.sum(alpha_0) < 0:
    log_file_path = os.path.join(PROJECT_PATH, 'results', '1a_error_dirichlet_score.log')
    log_file = open(log_file_path, 'w')
    log_file.write(str(alpha_0) + '\n')
    log_file.write(str(alpha_sum_approx) + '\n')
    log_file.write(str(observed_dirichlet.mean(axis=0)) + '\n')
    log_file.write(str(observed_dirichlet) + '\n')
    log_file.close()
    assert np.sum(alpha_0) < 0
  mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
  diri_score = dirichlet_normality_score(mle_alpha_t, x_eval_p)
  return diri_score

def correct_0_value_predictions(predictions):
  predictions[predictions==0] = 1e-10
  return normalize_sum1(predictions)

