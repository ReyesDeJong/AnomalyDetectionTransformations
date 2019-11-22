import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

"""
Original model, but only with translation transformations (9 transformations), original Resnet is used
"""

import numpy as np
import time
import datetime
from scripts.detached_transformer_od_hits import \
  plot_histogram_disc_loss_acc_thr
import matplotlib.pyplot as plt

data_path = os.path.join(
    PROJECT_PATH, 'results', 'Transformer_OD_Model',
    'ztf-real-bog-v1_Transformer-OD-Model-dirichlet_real_2019-11-22-0031.npz')
data = np.load(data_path)
data_dict = dict(data)

scores = data_dict['scores']
labels = data_dict['labels']
scores_val = data_dict['scores_val']
inliers_scores = scores[labels == 1]
outliers_scores = scores[labels != 1]

accuracies = data_dict['accuracies']
roc_thresholds = data_dict['roc_thresholds']


plot_histogram_disc_loss_acc_thr(inliers_scores, outliers_scores,
                                 path=None,
                                 x_label_name='Dscores',
                                 val_inliers_score=scores_val)

plt.plot(roc_thresholds, accuracies)
plt.xlim((scores.min(), scores.max()))
plt.show()


