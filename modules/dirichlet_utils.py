import os
import sys

import numpy as np
import tensorflow as tf

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from scipy.special import psi, polygamma


# ToDo: see if this can be done by tf

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

  log_p_hat_train = np.sum(np.log(observed_dirichlet), axis=0)
  alpha_sum_approx = calc_approx_alpha_sum(
      observed_dirichlet)
  alpha_0 = np.mean(observed_dirichlet, axis=0) * alpha_sum_approx
  mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
  diri_score = dirichlet_normality_score(mle_alpha_t, x_eval_p)
  return diri_score





def calc_approx_alpha_sum_tf(observations):
  N = observations.shape[0]
  f = tf.reduce_mean(observations, axis=0)
  numerator = (N * (f.shape[0] - 1.0) * (-tf.math.digamma(1.0)))
  denominator = (
      N * tf.reduce_sum(f * tf.math.log(f)) - tf.reduce_sum(
      f * tf.reduce_sum(tf.math.log(observations), axis=0)))
  # denominator = tf.dtypes.cast(denominator, tf.float32)
  return numerator / denominator


def inv_psi_tf(y, iters=5):
  # initial estimate
  cond = y >= -2.22
  cond = tf.dtypes.cast(cond, tf.float32)
  x = cond * (tf.math.exp(y) + 0.5) + (1.0 - cond) * -1.0 / (
      y - tf.math.digamma(1.0))

  for _ in range(iters):
    x = x - (tf.math.digamma(x) - y) / tf.math.polygamma(1.0, x)
  return x


def fixed_point_dirichlet_mle_tf(alpha_init, log_p_hat, max_iter=1000):
  alpha_new = alpha_old = alpha_init
  for _ in tf.range(max_iter):
    alpha_new = inv_psi_tf(
      tf.math.digamma(tf.reduce_sum(alpha_old)) + log_p_hat)
    if tf.sqrt(tf.reduce_sum((alpha_old - alpha_new) ** 2.0)) < 1e-9:
      break
    alpha_old = alpha_new
  return alpha_new


def dirichlet_normality_score_tf(alpha, p):
  return tf.reduce_sum((alpha - 1.0) * tf.math.log(p), axis=-1)

@tf.function
def dirichlet_score_tf(predict_x_train, predict_x_eval):
  observed_dirichlet = tf.dtypes.cast(predict_x_train, tf.float32)
  x_eval_p = tf.dtypes.cast(predict_x_eval, tf.float32)

  log_p_hat_train = tf.reduce_sum(tf.math.log(observed_dirichlet), axis=0)
  alpha_sum_approx = calc_approx_alpha_sum_tf(
      observed_dirichlet)
  # print(alpha_sum_approx)
  # print(tf.reduce_mean(observed_dirichlet, axis=0))
  alpha_0 = tf.reduce_mean(observed_dirichlet, axis=0) * alpha_sum_approx
  mle_alpha_t = fixed_point_dirichlet_mle_tf(alpha_0, log_p_hat_train)
  diri_score = dirichlet_normality_score_tf(mle_alpha_t, x_eval_p)
  return diri_score

if __name__ == '__main__':
  import time
  from modules import utils
  data = np.random.random((25000,72))
  start_time = time.time()
  pred_mat= dirichlet_score(data, data)
  print(
      "Time  dirichlet_score %s" % utils.timer(
          start_time, time.time()),
      flush=True)
  start_time = time.time()
  pred_mat = dirichlet_score_tf(data, data)
  print(
      "Time  dirichlet_score_tf %s" % utils.timer(
          start_time, time.time()),
      flush=True)