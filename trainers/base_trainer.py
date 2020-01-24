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
from modules.utils import create_auc_table


class Trainer(object):
  """
  Constructor
  """


  def __init__(self, data_loader: HiTSOutlierLoader,
      params={param_keys.RESULTS_FOLDER_NAME: '',
              param_keys.SCORES_TO_USE: None}):
    self.data_loader = data_loader
    self.all_models_metrics_message_dict = {}
    self.all_models_metrics_dict = {}
    self.print_manager = PrintManager()
    self.models_path = os.path.join(PROJECT_PATH, constants.RESULTS,
                                    params[param_keys.RESULTS_FOLDER_NAME])
    self.metrics_dict_template = {
      general_keys.ROC_AUC: {}, general_keys.ACC_AT_PERCENTIL: {},
      general_keys.MAX_ACCURACY: {}, general_keys.PR_AUC_NORM: {}}

  def append_model_metrics_to_all_it_models_metrics(self, it_metrics_dict,
      metric_dicts):
    for essential_metrics_keys in it_metrics_dict:
      for key in metric_dicts.keys():
        if key not in it_metrics_dict[essential_metrics_keys]:
          it_metrics_dict[essential_metrics_keys][key] = []
        it_metrics_dict[essential_metrics_keys][key].append(
            metric_dicts[key][essential_metrics_keys])

  def get_metrics_message(self, all_it_metrics, n_models_trained, model_name,
      transformer):
    message = '\n\n %i %s %s models %s Test' % (
      n_models_trained, transformer.name, model_name, self.data_loader.name)
    for essential_metrics_key in all_it_metrics:
      message += '\n\n  %s' % essential_metrics_key
      score_names = all_it_metrics[essential_metrics_key].keys()
      for score_key in score_names:
        mean = np.mean(
            all_it_metrics[essential_metrics_key][score_key])
        std = np.std(
            all_it_metrics[essential_metrics_key][score_key])
        message += '\n  %s : %.4f +/- %.4f' % (score_key, mean, std)
    return message

  def train_model_n_times(self, ModelClass,
      transformer: AbstractTransformer, params, train_times,
      model_name=None):
    self.data_loader = data_loader
    seed_array = np.arange(train_times).tolist()
    all_it_metrics = self.metrics_dict_template.copy()
    if training_data:
      (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = training_data
    else:
      (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = self.data_loader.get_outlier_detection_datasets()
    if model_name is None:
      aux_model = ModelClass(data_loader=self.data_loader,
                             transformer=transformer,
                             input_shape=x_train.shape[1:])
      model_name = aux_model.name
      del aux_model
    for i in range(len(seed_array)):
      model = ModelClass(data_loader=self.data_loader, transformer=transformer,
                         input_shape=x_train.shape[1:],
                         name=model_name, results_folder_name=self.models_path)
      model.fit(x_train, x_val, epochs=params['epochs'],
                patience=params['patience'])
      metrics_dict = model.evaluate_od(
          x_train, x_test, y_test, self.data_loader.name, general_keys.REAL,
          x_val, save_hist_folder_path=model.specific_model_folder)
      print('\nroc_auc')
      for key in metrics_dict.keys():
        print(key, metrics_dict[key]['roc_auc'])
      print('\nacc_at_percentil')
      for key in metrics_dict.keys():
        print(key, metrics_dict[key]['acc_at_percentil'])
      print('\nmax_accuracy')
      for key in metrics_dict.keys():
        print(key, metrics_dict[key]['max_accuracy'])
      self.append_model_metrics_to_all_it_models_metrics(all_it_metrics,
                                                         metrics_dict)
      printing_message = self.get_metrics_message(all_it_metrics, i + 1,
                                                  model_name, transformer)
      del model
    self.print_to_log(printing_message,
                      '%s_%s' % (self.data_loader.name, model_name))
    self.all_models_metrics_dict[model_name] = all_it_metrics
    self.all_models_metrics_message_dict[model_name] = printing_message
    return all_it_metrics

  def print_all_models_metrics(self):
    msg = ''
    for model_name in self.all_models_metrics_message_dict.keys():
      msg += self.all_models_metrics_message_dict[model_name]
    model_names = list(self.all_models_metrics_message_dict.keys())
    self.print_to_log(msg, self.data_loader.name + '_'.join(model_names))

  def print_to_log(self, msg, log_name):
    log_file = log_name + '.log'
    print = self.print_manager.verbose_printing(True)
    file = open(os.path.join(self.models_path, log_file), 'a')
    self.print_manager.file_printing(file)
    print(msg)
    self.print_manager.close()
    file.close()

  # TODO: remove hardcoding of folder name
  def create_tables_of_results_folders(self):
    models_folder_names = [dI for dI in os.listdir(self.models_path) if
                           os.path.isdir(os.path.join(self.models_path, dI))]
    for model_folder_name in models_folder_names:
      all_metric_files_paths = os.path.join(self.models_path, model_folder_name,
                                            'all_metric_files')
      metrics_names = self.metrics_dict_template.keys()
      for single_metric_name in metrics_names:
        create_auc_table(all_metric_files_paths, single_metric_name)

  def get_training_data(self, transformer: AbstractTransformer,
      data_loader: HiTSOutlierLoader):
    (x_train, y_train), (x_val, y_val), (
      x_test, y_test) = data_loader.get_outlier_detection_datasets()
    x_train_transform, y_train_transform = transformer.apply_all_transforms(
        x=x_train)
    x_val_transform, y_val_transform = transformer.apply_all_transforms(
        x=x_val)
    x_test_transform, y_test_transform = transformer.apply_all_transforms(
        x=x_test)
    training_data = (x_train_transform, y_train_transform), (
      x_val_transform, y_val_transform), (x_test_transform, y_test)
    return training_data


if __name__ == '__main__':
  from parameters import loader_keys
  from models.transformer_od_already_transformed import AlreadyTransformODModel
  # from models.transformer_od_simple_net import TransformODSimpleModel
  from modules.geometric_transform import transformations_tf
  import tensorflow as tf

  training_times = 10

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  train_params = {
    param_keys.RESULTS_FOLDER_NAME: 'Already_transformed',
    'epochs': 1,
    'patience': 0
  }
  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2, 3],  # [2],#
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  data_loader = HiTSOutlierLoader(hits_params)
  trainer = Trainer(data_loader, train_params)
  transformer = transformations_tf.Transformer()
  training_data = trainer.get_training_data(transformer, data_loader)
  trainer.train_model_n_times(
      AlreadyTransformODModel, transformer, train_params,
      train_times=training_times, training_data=training_data)

  trans_transformer = transformations_tf.TransTransformer
  training_data = trainer.get_training_data(trans_transformer, data_loader)
  trainer.train_model_n_times(
      AlreadyTransformODModel, trans_transformer, train_params,
      train_times=training_times, training_data=training_data)

  # trainer.train_model_n_times(TransformODSimpleModel, data_loader, transformer,
  #                             params, train_times=3)

  trainer.print_all_models_metrics()
