"""
Model that recives as input transformed data
"""

import os
import sys

PROJECT_PATH = \
  os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# os.path.abspath('./../../../home/ereyes/Projects/Thesis/AnomalyDetectionTransformations')
sys.path.append(PROJECT_PATH)
import tensorflow as tf
from modules.geometric_transform.transformations_tf import AbstractTransformer
from parameters import general_keys
import numpy as np
from sklearn.svm import OneClassSVM
from models.transformer_od import TransformODModel

"""In situ transformation perform"""


def _train_ocsvm_and_score(params, x_train, val_labels, x_val):
  return np.mean(val_labels ==
                 OneClassSVM(**params).fit(x_train).predict(
                     x_val))


# TODO: think if its better to create a trainer instead of an encapsulated model
class AlreadyTransformODModel(TransformODModel):
  def __init__(self, data_loader=None, transformer: AbstractTransformer=None, input_shape=None, depth=10,
      widen_factor=4, results_folder_name='',
      name='Already_Transformer_OD_Model', **kwargs):
    super().__init__(None,
                     transformer, input_shape, depth,
                     widen_factor, results_folder_name,
                     name, **kwargs)
    self.transformer.set_return_data_not_transformed(True)


if __name__ == '__main__':
  from parameters import loader_keys
  from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
  from modules.geometric_transform import transformations_tf
  import time
  from modules import utils

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  start_time = time.time()
  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2, 3],
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  hits_outlier_dataset = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = hits_outlier_dataset.get_outlier_detection_datasets()

  transformer = transformations_tf.Transformer()
  x_train_transform, y_train_transform = transformer.apply_all_transforms(
      x=x_train)
  x_val_transform, y_val_transform = transformer.apply_all_transforms(
      x=x_val)
  x_test_transform, y_test_transform = transformer.apply_all_transforms(
      x=x_test)

  mdl = AlreadyTransformODModel(transformer=transformer,
                                input_shape=x_train.shape[1:])

  batch_size = 128
  mdl.fit(
      x_train_transform, x_val_transform, train_batch_size=batch_size,
      epochs=2  # int(np.ceil(200 / transformer.n_transforms))
  )
  met_dict = mdl.evaluate_od(
      x_train_transform,  x_test_transform, y_test, 'hits-4-c', 'real',
      x_val_transform)

  print('\nroc_auc')
  for key in met_dict.keys():
    print(key, met_dict[key]['roc_auc'])
  print(
      "Train and evaluate %s" % utils.timer(
          start_time, time.time()))
  """
  roc_auc
  dirichlet 0.9896582500000001
  matrix_trace 0.9541035
  entropy 0.9820515000000001
  cross_entropy 0.9614397499999999
  mutual_information 0.9889197499999999
  
  tf2
  dirichlet 0.9871679999999999
  Train and evaluate 00:02:40.02
  Matrix_score_Time 00:00:12.85
  Not
  Appliying all 72 transforms to set of shape (288000, 21, 21, 4)
  (288768, 21, 21, 4)
  Matrix_score_Time 00:00:06.06
  100%|███████████████████████████████████████████| 72/72 [00:23<00:00,  3.07it/s]
  Not
  Appliying all 72 transforms to set of shape (72000, 21, 21, 4)
  (72704, 21, 21, 4)
  Matrix_score_Time 00:00:01.55
  100%|███████████████████████████████████████████| 72/72 [00:22<00:00,  3.15it/s]
  roc_auc
  dirichlet 0.76338925
  Train and evaluate 00:01:14.88



  keras
  dirichlet 0.9830505
  Train and evaluate 00:02:46.41
  train 00:01:37.39
  Matrix_score_Time 00:00:11.44
  Not
  Appliying all 72 transforms to set of shape (288000, 21, 21, 4)
  Matrix_score_Time 00:00:05.20
  100%|███████████████████████████████████████████| 72/72 [00:05<00:00, 12.66it/s]
  Not
  Appliying all 72 transforms to set of shape (72000, 21, 21, 4)
  Matrix_score_Time 00:00:01.33
  100%|███████████████████████████████████████████| 72/72 [00:05<00:00, 13.92it/s]
  roc_auc
  dirichlet 0.7630552500000001
  Train and evaluate 00:00:36.91


  """
