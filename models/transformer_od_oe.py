"""
First attempt (train_step_tf2 like) to build geometric trasnformer for outlier detection in tf2
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import tensorflow as tf
from modules.networks.train_step_tf2.wide_residual_network_oe import \
  WideResidualNetworkOE
from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from parameters import general_keys
import numpy as np
from modules import dirichlet_utils, utils
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from modules.metrics import accuracies_by_threshold, accuracy_at_thr
import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed
from sklearn.svm import OneClassSVM
import time
from models.transformer_od import TransformODModel


# TODO: think if its better to create a trainer instead of an encapsulated model
class TransformODModelOE(TransformODModel):
  def __init__(self, data_loader: ZTFOutlierLoader,
      transformer: AbstractTransformer, input_shape, depth=10,
      widen_factor=4, results_folder_name='', name='Transformer_OD_Model_OE',
      **kwargs):
    super().__init__(data_loader,
                     transformer, input_shape, depth,
                     widen_factor, results_folder_name, name)

  def get_network(self, input_shape, n_classes,
      depth, widen_factor, model_path, **kwargs):
    return WideResidualNetworkOE(
        input_shape=input_shape, n_classes=n_classes, depth=depth,
        widen_factor=widen_factor, model_path=model_path, **kwargs)

  # def fit(self, x_train, x_train_oe, x_val, transform_batch_size=512,
  #     train_batch_size=128,
  #     epochs=2, patience=0, **kwargs):
  #   if epochs is None:
  #     epochs = int(np.ceil(200 / self.transformer.n_transforms))
  #   # ToDo: must be network? or just self.compile???
  #   self.network.compile(
  #       general_keys.ADAM, general_keys.CATEGORICAL_CROSSENTROPY,
  #       [general_keys.ACC])
  #   x_train_transform, y_train_transform = \
  #     self.transformer.apply_all_transforms(
  #         x=x_train, batch_size=transform_batch_size)
  #   x_train_oe_transform, y_train_oe_transform = \
  #     self.transformer.apply_all_transforms(
  #         x=x_train_oe, batch_size=transform_batch_size)
  #   y_train_oe_transform = y_train_oe_transform * 0 - 99
  #   print(y_train_oe_transform)
  #   x_val_transform, y_val_transform = \
  #     self.transformer.apply_all_transforms(
  #         x=x_val, batch_size=transform_batch_size)
  #   es = tf.keras.callbacks.EarlyStopping(
  #       monitor='val_loss', mode='min', verbose=1, patience=patience,
  #       restore_best_weights=True)
  #   if epochs < 3 or epochs == int(
  #       np.ceil(200 / self.transformer.n_transforms)):
  #     es = tf.keras.callbacks.EarlyStopping(
  #         monitor='val_loss', mode='min', verbose=1, patience=1e100,
  #         restore_best_weights=False)
  #   # del x_train_oe, y_train_oe_transform, x_train_oe_transform, x_train, x_train_transform
  #   self.network.fit(
  #       x=x_train_transform, y=y_train_transform,
  #       validation_data=(
  #         x_val_transform, y_val_transform),
  #       batch_size=train_batch_size,
  #       epochs=epochs, callbacks=[es], **kwargs)
  #   weight_path = os.path.join(self.checkpoint_folder,
  #                              'final_weights.ckpt')
  #   del x_val, x_val_transform, y_train_transform, y_val_transform
  #   self.save_weights(weight_path)


  def fit(self, x_train, x_train_oe, x_val, transform_batch_size=512,
      train_batch_size=128,
      epochs=2, patience=0, **kwargs):
    if epochs is None:
      epochs = int(np.ceil(200 / self.transformer.n_transforms))
    # ToDo: must be network? or just self.compile???
    self.network.compile(
        general_keys.ADAM, general_keys.CATEGORICAL_CROSSENTROPY,
        [general_keys.ACC])
    x_train_transform, y_train_transform = \
      self.transformer.apply_all_transforms(
          x=x_train, batch_size=transform_batch_size)

    x_train_oe_transform, y_train_oe_transform = \
      self.transformer.apply_all_transforms(
          x=x_train_oe, batch_size=transform_batch_size)
    y_train_oe_transform = y_train_oe_transform * 0 - 99
    print(y_train_oe_transform)

    x_val_transform, y_val_transform = \
      self.transformer.apply_all_transforms(
          x=x_val, batch_size=transform_batch_size)
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=patience,
        restore_best_weights=True)
    if epochs < 3 or epochs == int(
        np.ceil(200 / self.transformer.n_transforms)):
      es = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss', mode='min', verbose=1, patience=1e100,
          restore_best_weights=False)

    x_train_merge = np.concatenate([x_train_transform, x_train_oe_transform], axis=0)
    y_train_merge = np.concatenate([y_train_transform, y_train_oe_transform], axis=0)
    idxes = np.arange(len(y_train_merge))
    np.random.RandomState(42).shuffle(idxes)
    x_train_merge = x_train_merge[idxes]
    y_train_merge = y_train_merge[idxes]
    self.network.fit(
        x=x_train_merge, y=y_train_merge,
        validation_data=(
          x_val_transform, y_val_transform),
        batch_size=train_batch_size,
        epochs=epochs, callbacks=[es], **kwargs)
    # self.network.eval_tf(x_val_transform, tf.keras.utils.to_categorical(y_val_transform))
    weight_path = os.path.join(self.checkpoint_folder,
                               'final_weights.ckpt')
    del x_train, x_val, x_train_transform, x_val_transform, y_train_transform, y_val_transform
    self.save_weights(weight_path)