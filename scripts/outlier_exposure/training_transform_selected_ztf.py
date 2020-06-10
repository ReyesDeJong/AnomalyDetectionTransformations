"""
Just replicate results near of paper
"""


import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform import transformations_tf
from modules.geometric_transform.transformer_for_ranking_old_bugged import \
  RankingTransformerOld
from models.transformer_od import TransformODModel
from models.transformer_od import TransformODModel
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
from modules.geometric_transform.transformations_tf import Transformer, \
  TransTransformer
from itertools import chain, combinations
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys
import numpy as np
from modules.utils import save_pickle

if __name__ == "__main__":
  MODEL_CHKP_PATH = os.path.join(PROJECT_PATH, 'results', 'OutlierExposure', 'ZTF_TRF_SEL_TEST')
  N_RUNS = 1

  transformer = transformations_tf.PlusKernelTransformer()
  new_transformation_tuples_list = (
    [(0, 0, 0, 0, 0, 0), (0, 0, -8, 0, 0, 0),
     (0, 0, 8, 0, 0, 0), (0, -8, 0, 0, 0, 0),
     (0, -8, -8, 0, 0, 0), (0, -8, 8, 0, 0, 0),
     (0, 8, 0, 0, 0, 0), (0, 8, -8, 0, 0, 0),
     (0, 8, 8, 0, 0, 0),

     (0, 0, 0, 0, 1, 0), (0, 8, -8, 0, 1, 0),

     (0, 0, 0, 0, 0, 1), (0, 0, -8, 0, 0, 1),
     (0, 0, 8, 0, 0, 1), (0, -8, 0, 0, 0, 1),
     (0, -8, -8, 0, 0, 1), (0, -8, 8, 0, 0, 1),
     (0, 8, 0, 0, 0, 1), (0, 8, -8, 0, 0, 1),
     (0, 8, 8, 0, 0, 1),

     (0, 0, 0, 0, 1, 1),
     (0, 0, -8, 0, 1, 1), (0, 0, 8, 0, 1, 1),
     (0, -8, 0, 0, 1, 1), (0, -8, -8, 0, 1, 1),
     (0, -8, 8, 0, 1, 1), (0, 8, 0, 0, 1, 1),
     (0, 8, -8, 0, 1, 1), (0, 8, 8, 0, 1, 1)
     ])
  new_transforms = new_transformation_tuples_list
  print('Initial N Transformations: ', transformer.n_transforms)
  transformer.set_transformations_to_perform(new_transforms)
  print('Final N Transformations: ', transformer.n_transforms)
  print(transformer.transformation_tuples)
  transformer.name = '%s_after_selection_%i' % (
    transformer.name, int(transformer.n_transforms))

  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  hits_loader = HiTSOutlierLoader(hits_params)
  ztf_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH,
        '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
  }
  ztf_loader = ZTFSmallOutlierLoader(ztf_params)

  data_loaders = (hits_loader, ztf_loader)
  loader_i = data_loaders[1]
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = loader_i.get_outlier_detection_datasets()

  model = TransformODModel(
              loader_i, transformer, input_shape=x_train.shape[1:],
              results_folder_name=MODEL_CHKP_PATH)
  model.fit(x_train, x_val, epochs=1000, patience=0)
  results = model.evaluate_od(x_train, x_test,
                              y_test, loader_i.name,
                              'real', x_val)
  print(results['dirichlet']['roc_auc'])