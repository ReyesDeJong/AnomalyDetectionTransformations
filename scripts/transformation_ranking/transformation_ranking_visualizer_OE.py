"""
Training a model with basic-non-composed transforms, to visualize it feature
 layer and
then calculate FID
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
import pandas as pd

if __name__ == "__main__":

  # results_all_runs = pd.read_pickle('aux_results/small_rank_small_ztf.pkl')
  results_all_runs = pd.read_pickle('aux_results/small_rank_hits_4_channels.pkl')
  n_runs = list(results_all_runs.keys())
  trf_idxs = list(results_all_runs[0].keys())
  outlier_types = results_all_runs[0][0].keys()
  print(outlier_types)
  outlier_to_see = 'gt'
  # outlier_to_see = 'other_astro'
  # outlier_to_see = 'cifar10'
  # outlier_to_see = 'shuffle'
  # outlier_to_see = 'mnist'
  for trf_i in trf_idxs:
    print(len(results_all_runs[0][trf_i][outlier_to_see][0]),
          results_all_runs[0][trf_i][outlier_to_see][1]['dirichlet']['roc_auc'])

  key_to_get = 'roc_auc'
  # key_to_get = 'pr_auc_anom'
  # key_to_get = 'acc_at_percentil'
  results_stats = {}

  for trf_idx_i in trf_idxs:
    results_stats[trf_idx_i] = [results_all_runs[0][trf_idx_i][outlier_to_see][0], []]
    for run_i in n_runs:
      results_stats[trf_idx_i][1].append(
          results_all_runs[run_i][trf_idx_i][outlier_to_see][1]['dirichlet'][key_to_get])

  means = []
  stds = []
  trfs_list = []
  trf_idx_list = []
  for trf_idx_i in results_stats.keys():
    trf_idx_list.append(trf_idx_i)
    mean_i =np.mean(results_stats[trf_idx_i][1])
    std_i = np.std(results_stats[trf_idx_i][1])
    trf_i = results_stats[trf_idx_i][0]
    means.append(mean_i)
    stds.append(std_i)
    trfs_list.append(trf_i)

  sort_idxs = np.argsort(means)
  sort_means = np.array(means)[sort_idxs]
  sort_stds = np.array(stds)[sort_idxs]
  sort_trf_list = np.array(trfs_list)[sort_idxs]
  sort_trf_idx_list = np.array(trf_idx_list)[sort_idxs]

  for i in range(len(sort_trf_list)):
    print(
        sort_trf_list[i],
            'len %i: %.4f+/-%.4f' % (len(sort_trf_list[i]), sort_means[i], sort_stds[i]))

  print('\nWORST N')
  for i in list(range(len(sort_trf_list)))[:10]:
    print(
        sort_trf_list[i],
            'idx %i len %i: %.4f+/-%.4f' % (sort_trf_idx_list[i], len(sort_trf_list[i]), sort_means[i], sort_stds[i]))


  print('\nBEST N')
  for i in list(range(len(sort_trf_list)))[-10:]:
    print(
        sort_trf_list[i],
            'idx %i len %i: %.4f+/-%.4f' % (sort_trf_idx_list[i], len(sort_trf_list[i]), sort_means[i], sort_stds[i]))
