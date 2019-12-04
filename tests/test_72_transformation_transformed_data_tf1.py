import os
import sys

"""
Test 72 transform, if transformed data by old transformer is the same in tf2 
and tf1

Conclusion: It is the same
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import matplotlib;

matplotlib.use('agg')
import pandas as pd
import tensorflow as tf
import numpy as np
import transformations


def test_transformed_data_tf1_tf2(transformer,
    normal_data_name='tf2_normal.pkl',
    tf2_transformed_data_name='tf2_old_transformed.pkl'):
  save_dir = os.path.join(PROJECT_PATH, 'tests', 'aux_data')
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = pd.read_pickle(os.path.join(save_dir, normal_data_name))
  y_train_transform_tf1 = np.tile(np.arange(transformer.n_transforms),
                                 len(x_train))
  x_train_transform_tf1 = transformer.transform_batch(
    np.repeat(x_train, transformer.n_transforms, axis=0),
    y_train_transform_tf1)
  y_val_transform_tf1 = np.tile(np.arange(transformer.n_transforms),
                                 len(x_val))
  x_val_transform_tf1 = transformer.transform_batch(
    np.repeat(x_val, transformer.n_transforms, axis=0),
    y_val_transform_tf1)
  y_test_transform_tf1 = np.tile(np.arange(transformer.n_transforms),
                                 len(x_test))
  x_test_transform_tf1 = transformer.transform_batch(
    np.repeat(x_test, transformer.n_transforms, axis=0),
    y_test_transform_tf1)
  (x_train_transform_tf2, y_train_transform_tf2), (
    x_val_transform_tf2, y_val_transform_tf2), (
    x_test_transform_tf2, y_test_transform_tf2) = pd.read_pickle(
      os.path.join(save_dir, tf2_transformed_data_name))

  print(np.mean(x_train_transform_tf1==x_train_transform_tf2))
  print(np.mean(y_train_transform_tf1 == y_train_transform_tf2))
  print(np.mean(x_test_transform_tf1 == x_test_transform_tf2))
  print(np.mean(y_test_transform_tf1 == y_test_transform_tf2))
  print(np.mean(x_val_transform_tf1 == x_val_transform_tf2))
  print(np.mean(y_val_transform_tf1 == y_val_transform_tf2))

  return (x_train_transform_tf1, y_train_transform_tf1, x_val_transform_tf1,
          y_val_transform_tf1, x_test_transform_tf1, y_test_transform_tf1), (
         x_train_transform_tf2, y_train_transform_tf2, x_val_transform_tf2,
         y_val_transform_tf2, x_test_transform_tf2, y_test_transform_tf2)


if __name__ == '__main__':
  results_path = os.path.join(PROJECT_PATH, 'results', 'replication')
  # Not use all gpu
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  # # Transformer
  transformer_old = transformations.Transformer()
  tf1_data, tf2_data = test_transformed_data_tf1_tf2(transformer_old)
  for i in range(len(tf1_data)):
    print(np.mean(tf1_data[i] == tf2_data[i]))
