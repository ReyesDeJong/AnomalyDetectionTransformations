import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

import csv
from collections import defaultdict
from glob import glob
import numpy as np
import scipy.stats
from modules import utils
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from modules.geometric_transform.transformations_tf import AbstractTransformer
from models.transformer_od import TransformODModel

RESULTS_DIR = os.path.join(PROJECT_PATH, 'results/ztf-refact')


# TODO: construct evaluator to only perfor new metrics calculation
# TODO: make abstract data loader
def _transformations_experiment(data_loader: ZTFOutlierLoader,
    transformer: AbstractTransformer, dataset_name: str, class_name: str,
    save_path: str):
  # Todo: pass as param_dict
  batch_size = 128
  transform_batch_size = 1024

  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader.get_outlier_detection_datasets()

  mdl = TransformODModel(
      data_loader=data_loader, transformer=transformer,
      input_shape=x_train.shape[1:], results_folder_name=save_path)

  mdl.fit(x=x_train, transform_batch_size=transform_batch_size,
          train_batch_size=batch_size,
          epochs=2  # int(np.ceil(200 / transformer.n_transforms))
          )

  metrics_dict = mdl.evaluate_od(
      x_train, x_test, y_test, dataset_name, class_name, x_val,
      transform_batch_size=transform_batch_size,
      additional_save_path_list=save_path)
  del mdl


# ToDo: research how to perform multi gpu training
def run_experiments(data_loader, transformer, dataset_name, class_name, n_runs):
  save_path = os.path.join(RESULTS_DIR, dataset_name)
  utils.check_paths(save_path)

  # Transformations
  for _ in range(n_runs):
    _transformations_experiment(data_loader, transformer, dataset_name,
                                class_name, save_path)


def create_auc_table(metric='roc_auc'):
  file_path = glob(os.path.join(RESULTS_DIR, '*', '*.npz'))
  results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  methods = set()
  for p in file_path:
    _, f_name = os.path.split(p)
    dataset_name, method, single_class_name = f_name.split(sep='_')[:3]
    methods.add(method)
    npz = np.load(p)
    roc_auc = npz[metric]
    results[dataset_name][single_class_name][method].append(roc_auc)

  for ds_name in results:
    for sc_name in results[ds_name]:
      for method_name in results[ds_name][sc_name]:
        roc_aucs = results[ds_name][sc_name][method_name]
        print(method_name, ' ', roc_aucs)
        results[ds_name][sc_name][method_name] = [np.mean(roc_aucs),
                                                  0 if len(
                                                      roc_aucs) == 1 else scipy.stats.sem(
                                                      np.array(roc_aucs))
                                                  ]

  with open(os.path.join(RESULTS_DIR, 'results-{}.csv'.format(metric)),
            'w') as csvfile:
    fieldnames = ['dataset', 'single class name'] + sorted(list(methods))
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ds_name in sorted(results.keys()):
      for sc_name in sorted(results[ds_name].keys()):
        row_dict = {'dataset': ds_name, 'single class name': sc_name}
        row_dict.update({method_name: '{:.5f} ({:.5f})'.format(
            *results[ds_name][sc_name][method_name])
          for method_name in results[ds_name][sc_name]})
        writer.writerow(row_dict)


if __name__ == '__main__':
  from parameters import loader_keys, general_keys
  import time

  from modules.geometric_transform.transformations_tf import Transformer, \
    TransTransformer

  N_RUNS = 2
  params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/ztf_v1_bogus_added.pkl'),
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2],
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  ztf_outlier_dataset = ZTFOutlierLoader(params)
  params[loader_keys.CROP_SIZE] = None
  ztf_outlier_dataset_63 = ZTFOutlierLoader(params)
  transformer = Transformer()
  trans_transformer = TransTransformer()
  # data_loader, transformer, dataset_name, class_idx_to_run_experiments_on, n_runs
  #TODO: delgate names to data_laoders
  experiments_list = [
    (
      ztf_outlier_dataset, transformer, 'ztf-real-bog-v1-refact', 'real',
      N_RUNS),
    (
      ztf_outlier_dataset, transformer, 'ztf-real-bog-v1-refact', 'real',
      N_RUNS),
    (ztf_outlier_dataset_63, trans_transformer, 'ztf-real-bog-v1-63-refact',
     'real', N_RUNS),
    (ztf_outlier_dataset_63, trans_transformer, 'ztf-real-bog-v1-63-refact',
     'real', N_RUNS),
  ]
  start_time = time.time()
  for data_loader, transformer, dataset_name, class_name, run_i in experiments_list:
    run_experiments(data_loader, transformer, dataset_name, class_name, run_i)
  print(
      "Time elapsed to train everything: " + utils.timer(start_time,
                                                         time.time()))

  # metrics_to_create_table = {}
  create_auc_table()