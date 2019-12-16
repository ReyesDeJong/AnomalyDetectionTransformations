Learn
more or give
us
feedback
import os
import sys

import numpy as np

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.geometric_transform.transformations_tf import AbstractTransformer
from parameters import param_keys, general_keys, constants
from modules.print_manager import PrintManager


class Trainer(object):
  """
  Constructor
  """

  def __init__(self, params={param_keys.RESULTS_FOLDER_NAME: ''}):
    self.all_models_metrics_dict = {}
    self.print_manager = PrintManager()
    self.model_path = os.path.join(PROJECT_PATH, constants.RESULTS,
                                   params[param_keys.RESULTS_FOLDER_NAME])

  def append_model_metrics_to_all_it_models_metrics(self, it_metrics_dict,
      metric_dicts):
    for essential_metrics_keys in it_metrics_dict:
      for key in metric_dicts.keys():
        if key not in it_metrics_dict[essential_metrics_keys]:
          it_metrics_dict[essential_metrics_keys][key] = []
        it_metrics_dict[essential_metrics_keys][key].append(
            metric_dicts[key][essential_metrics_keys])

  def get_metrics_message(self, all_it_metrics, n_models_trained, model_name):
    message = '\n %i %s models Test\n' % (n_models_trained, model_name)
    for essential_metrics_key in all_it_metrics:
      message += '\n  %s' % essential_metrics_key
      score_names = all_it_metrics[essential_metrics_key].keys()
      for score_key in score_names:
        mean = np.mean(
            all_it_metrics[essential_metrics_key][score_key])
        std = np.std(
            all_it_metrics[essential_metrics_key][score_key])
        message += '\n  %s : %.4f +/- %.4f' % (score_key, mean, std)
    return message

  def train_model_n_times(self, ModelClass, data_loader: HiTSOutlierLoader,
      transformer: AbstractTransformer, params, train_times,
      model_name=None):
    seed_array = np.arange(train_times).tolist()
    all_it_metrics = {
      general_keys.ROC_AUC: {}, general_keys.ACC_AT_PERCENTIL: {},
      general_keys.MAX_ACCURACY: {}, general_keys.PR_AUC_NORM: {}}
    for i in range(len(seed_array)):
      if model_name is None:
        aux_model = ModelClass(params)
        model_name = aux_model.model_name
        del aux_model
      model = ModelClass(**params, name=model_name + '_%i' % i)
      data_loader = HiTSOutlierLoader(
          data_loader=data_loader, transformer=transformer,
          input_shape=x_train.shape[1:], **params)
      (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = data_loader.get_outlier_detection_datasets()
      model.fit(x_train, x_val, **params)
      metrics_dict = model.evaluate_od(
          x_train, x_test, y_test, data_loader.name, general_keys.REAL, x_val)
      self.append_model_metrics_to_all_it_models_metrics(all_it_metrics,
                                                         metrics_dict)
      printing_message = self.get_metrics_message(all_it_metrics, i + 1,
                                                  model_name)
      self.print_to_log(printing_message, model_name)
      self.all_models_acc[model_name] = {printing_message}
    return all_it_metrics

  def print_all_accuracies(self):
    msg = ''
    for model_name in self.all_models_acc.keys():
      msg += self.all_models_acc[model_name]
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
