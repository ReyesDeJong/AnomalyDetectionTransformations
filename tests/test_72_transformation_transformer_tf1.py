import os
import sys

"""
Test 72 transform tf2 model on hits
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from modules import utils
from modules.data_loaders.base_line_loaders import load_hits4c
import time
import tensorflow as tf
import transformations
import numpy as np


def transformer_test(transformer):

  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = load_hits4c(return_val=True)
  y_train_transform = np.tile(np.arange(transformer.n_transforms),
                                 len(x_train))
  x_train_transform = transformer.transform_batch(
    np.repeat(x_train, transformer.n_transforms, axis=0),
    y_train_transform)
  y_val_transform = np.tile(np.arange(transformer.n_transforms),
                                 len(x_val))
  x_val_transform = transformer.transform_batch(
    np.repeat(x_val, transformer.n_transforms, axis=0),
    y_val_transform)
  y_test_transform = np.tile(np.arange(transformer.n_transforms),
                                 len(x_test))
  x_test_transform = transformer.transform_batch(
    np.repeat(x_test, transformer.n_transforms, axis=0),
    y_test_transform)
  return x_train_transform, y_train_transform, x_val_transform, \
         y_val_transform, x_test_transform, y_test_transform

def transformer_traditional_test():
  # Transformer
  transformer = transformations.Transformer()
  start_time = time.time()
  _ = transformer_test(transformer)
  print(
      "Time transformer_test(transformer) %s" % utils.timer(
          start_time, time.time()),
      flush=True)
  """
  ---Running this code outside function----
  Initial RAM 5.8GB, FINAL 16.3GB, MAX ~20GB
  Time transformer_test(transformer) 00:03:57.47
  It accumulates 6GB of RAM at each run

  If transformer exist as an object, it use memory every time is used,
   deleting object dont free memory

  ---Running this function----
    Initial RAM 5.8GB, FINAL 6.6GB, MAX ~19.5GB
  Time transformer_test(transformer) 00:03:57.47
  It accumulates 0GB of RAM at each run
  """


if __name__ == '__main__':
  results_path = os.path.join(PROJECT_PATH, 'results', 'replication')

  # # Transformer
  # transformer_traditional_test()
  # Transformer
  transformer = transformations.Transformer()
  start_time = time.time()
  _ = transformer_test(transformer)
  print(
      "Time transformer_test(transformer) %s" % utils.timer(
          start_time, time.time()),
      flush=True)

