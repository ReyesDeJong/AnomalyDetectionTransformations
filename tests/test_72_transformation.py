import os
import sys

"""
Test 72 transform tf2 model on hits
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import matplotlib;

matplotlib.use('agg')
from modules import utils
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys
from modules.geometric_transform import transformations_tf
import time
import tensorflow as tf


def transformer_test(transformer):
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
  x_train_transform, y_train_transform = transformer.apply_all_transforms(
      x=x_train)
  x_val_transform, y_val_transform = transformer.apply_all_transforms(
      x=x_val)
  x_test_transform, y_test_transform = transformer.apply_all_transforms(
      x=x_test)
  return x_train_transform, y_train_transform, x_val_transform, \
         y_val_transform, x_test_transform, y_test_transform


def transformer_tf_test():
  # Transformer tf
  transformer_tf = transformations_tf.Transformer()
  start_time = time.time()
  _ = transformer_test(transformer_tf)
  print(
      "Time transformer_test(transformer_tf) %s" % utils.timer(
          start_time, time.time()),
      flush=True)
  """
  Initial RAM 4.6GB, FINAL 13.2GB, MAX ~13.2GB
  Time transformer_test(transformer_tf) 00:00:08.29
  It accumulates 6GB of RAM at each run
  If transformer exist as an object, it use memory every time is used,
   deleting object dont free memory
  """


if __name__ == '__main__':
  results_path = os.path.join(PROJECT_PATH, 'results', 'replication')
  # Not use all gpu
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  # transformer_tf_test()

  # # Transformer
  # transformer = transformations.Transformer()
  # start_time = time.time()
  # _ = transformer_test(transformer)
  # print(
  #     "Time transformer_test(transformer) %s" % utils.timer(
  #         start_time, time.time()),
  #     flush=True)
  # """
  # Initial RAM 5.37GB, FINAL 11.4GB, MAX ~13.3GB
  # Time transformer_test(transformer) 00:02:24.06
  # It accumulates 6GB of RAM at each run
  # """

  # # Data loading
  # hits_params = {
  #   loader_keys.DATA_PATH: os.path.join(
  #       PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
  #   loader_keys.N_SAMPLES_BY_CLASS: 10000,
  #   loader_keys.TEST_PERCENTAGE: 0.2,
  #   loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
  #   loader_keys.USED_CHANNELS: [0, 1, 2, 3],
  #   loader_keys.CROP_SIZE: 21,
  #   general_keys.RANDOM_SEED: 42,
  #   loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  # }
  # hits_outlier_dataset = HiTSOutlierLoader(hits_params)
  # (x_train, y_train), (x_val, y_val), (
  #   x_test, y_test) = hits_outlier_dataset.get_outlier_detection_datasets()

  # # model
  # model = TransformODModel(
  #     data_loader=hits_outlier_dataset, transformer=transformer_tf,
  #     input_shape=x_train.shape[1:], results_folder_name=results_path)
  # # training
  #
  # start_time = time.time()
  # model.fit(x_train, x_val, transform_batch_size=1024,
  #           epochs=int(np.ceil(200 / transformer_tf.n_transforms)))
  # print(
  #     "Time model.fit %s" % utils.timer(
  #         start_time, time.time()),
  #     flush=True)
  #
  # start_time = time.time()
  # met_dict = model.evaluate_od(
  #     x_train, x_test, y_test, 'ztf-real-bog-v1', 'real', x_val)
  # print(
  #     "Time model.evaluate_od %s" % utils.timer(
  #         start_time, time.time()),
  #     flush=True)
