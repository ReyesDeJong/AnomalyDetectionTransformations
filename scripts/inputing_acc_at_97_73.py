import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import tensorflow as tf
from modules.networks.train_step_tf2.wide_residual_network import \
  WideResidualNetwork
# from modules.networks.wide_residual_network import WideResidualNetwork
from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from parameters import general_keys
import numpy as np
from modules import dirichlet_utils, utils
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from modules.metrics import accuracies_by_threshold, accuracy_at_thr
import pprint
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed
from sklearn.svm import OneClassSVM
import matplotlib
import time

matplotlib.use('Agg')

path_to_dicts = os.path.join(
    PROJECT_PATH,
    'results/PAPER_RESULTS/HITS_NAIVE_TRANSFORMS_D_to_have_percentile_97_73/resnet_VAL_epochs/Transformer_OD_Model/all_metric_files/hits_4_channels'
    # 'results/PAPER_RESULTS/ZTF_NAIVE_TRANSFORMS_E_to_have_percentile_97_73/resnet_VAL_epochs/Transformer_OD_Model/all_metric_files/small_ztf'
)

from os import listdir
from os.path import isfile, join
mypath = path_to_dicts
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
print(len(onlyfiles))


def input_97_73_score(dicts_folde, dict_name):
  percentile = 97.73
  dict_path = os.path.join(dicts_folde, dict_name)
  data_dict = dict(np.load(dict_path))
  #scores_val = score_metric_dict['scores_val']
  scores = data_dict['scores']
  labels = data_dict['labels']
  scores_val = data_dict['scores_val']
  scores_pos = scores[labels == 1]
  scores_neg = scores[labels != 1]
  truth = np.concatenate(
      (np.zeros_like(scores_neg), np.ones_like(scores_pos)))
  preds = np.concatenate((scores_neg, scores_pos))
  fpr, tpr, roc_thresholds = roc_curve(truth, preds)
  roc_auc = auc(fpr, tpr)
  accuracies = accuracies_by_threshold(labels, scores, roc_thresholds)
  # 100-percentile is necesary because normal data is at the right of anormal
  thr = np.percentile(scores_val, 100 - percentile)
  acc_at_percentil = accuracy_at_thr(labels, scores, thr)
  # pr curve where "normal" is the positive class
  precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(
      truth, preds)
  pr_auc_norm = auc(recall_norm, precision_norm)
  # pr curve where "anomaly" is the positive class
  precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(
      truth, -preds, pos_label=0)
  pr_auc_anom = auc(recall_anom, precision_anom)
  data_dict.update({'acc_at_percentil_97_73': acc_at_percentil})
  pprint.pprint(data_dict)
  np.savez_compressed(dict_path, **data_dict)

for name in onlyfiles:
  input_97_73_score(path_to_dicts, name)


