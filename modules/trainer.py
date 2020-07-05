"""
Train a model N times and return a specified metric on test set
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform.transformer_for_ranking import \
    RankingTransformer
from models.transformer_od_simple_net import TransformODSimpleModel
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
from itertools import chain, combinations
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys
import numpy as np
from modules import utils
import tensorflow as tf



class Trainer(object):
  """
  Constructor
  """

  def __init__(self, params={param_keys.RESULTS_FOLDER_NAME: ''}):
    self.all_models_acc = {}
    self.print_manager = PrintManager()
    self.model_path = os.path.join(PROJECT_PATH, 'results',
                                   params[param_keys.RESULTS_FOLDER_NAME])

  def train_model_n_times(self, ModelClass, params, train_times,
      model_name=None):
    seed_array = np.arange(train_times).tolist()
    accuracies = []
    for i in range(len(seed_array)):
      if model_name is None:
        aux_model = ModelClass(params)
        model_name = aux_model.model_name
        aux_model.close()
      model = ModelClass(params, model_name + '_%i' % i)
      metrics = model.fit()
      model.close()
      accuracies.append(metrics[general_keys.ACCURACY])
    self.print_to_log('\n %i %s models Test Accuracy: %.4f +/- %.4f' %
                      (len(seed_array), model_name, np.mean(accuracies),
                       np.std(accuracies)), model_name)
    self.all_models_acc[model_name] = {general_keys.MEAN: np.mean(accuracies),
                                       general_keys.STD: np.std(accuracies)}
    return accuracies

  def print_all_accuracies(self):
    msg = ''
    for model_name in self.all_models_acc.keys():
      model_metrics = self.all_models_acc[model_name]
      msg += "\n %s Test Accuracy: %.4f +/- %.4f" % \
             (model_name, model_metrics[general_keys.MEAN],
              model_metrics[general_keys.STD])
    model_names = list(self.all_models_acc.keys())
    self.print_to_log(msg, '_'.join(model_names))

  def print_to_log(self, msg, log_name):
    log_file = log_name + '.log'
    print = self.print_manager.verbose_printing(True)
    file = open(os.path.join(self.model_path, log_file), 'a')
    self.print_manager.file_printing(file)
    print(msg)
    self.print_manager.close()
    file.close()
