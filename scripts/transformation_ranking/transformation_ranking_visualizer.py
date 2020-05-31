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

import pandas as pd

if __name__ == "__main__":
  results_all_runs = pd.read_pickle('rank_small_ztf.pkl')
  # results_all_runs = pd.read_pickle('rank_hits_4_channels.pkl')
  n_runs = list(results_all_runs.keys())
  trf_idxs = list(results_all_runs[0].keys())
  for trf_i in trf_idxs:
    print(len(results_all_runs[0][trf_i][0]), results_all_runs[0][trf_i][1]['dirichlet']['roc_auc'])