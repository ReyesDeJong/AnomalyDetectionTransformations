import numpy as np
from scipy.special import psi
import os, sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

def calc_approx_alpha_sum(observations):
  N = len(observations)
  f = np.mean(observations, axis=0)

  return (N * (len(f) - 1) * (-psi(1))) / (
      N * np.sum(f * np.log(f)) - np.sum(
      f * np.sum(np.log(observations), axis=0)))

def normalize_sum1(array, axis=-1):
  sums = np.sum(array, axis=axis)
  return array / np.expand_dims(sums, axis)

log_file_folder = os.path.join(PROJECT_PATH, 'results', 'error_log')
bug_array = np.load(os.path.join(log_file_folder, 'observation.npy'))
observations = bug_array
alpha_bug = calc_approx_alpha_sum(bug_array)
# print(alpha_bug)
n_a = 10000
n_pred = 9
a = np.random.random((n_a,n_pred))
a = np.zeros_like(a)
random_ints = np.random.randint(0, high=n_pred, size=n_a)
# random_ints = np.array(list(zip(np.arange(n_a),random_ints)))
# a[random_ints] = 10
a[np.arange(n_a), random_ints] = 10000
# print(a)
a[a==0] = 1e-10
a_norm = normalize_sum1(a)
# print(a_norm)
# print(np.sum(a_norm,axis=-1))
alpha = calc_approx_alpha_sum(a_norm)
print(alpha)