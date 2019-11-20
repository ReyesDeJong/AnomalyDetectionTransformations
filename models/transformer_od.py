"""
First attempt (keras like) to build geometric trasnformer for outlier detection in tf2
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import tensorflow as tf
from modules.networks.wide_residual_network import WideResidualNetwork
from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from parameters import general_keys

"""In situ transformation perform"""


# TODO: think if its better to create a trainer instead of an encapsulated model
class TransformODModel(tf.keras.Model):
  def __init__(
      self, data_loader: ZTFOutlierLoader, transformer: AbstractTransformer,
      input_shape, n_classes, depth=10,
      widen_factor=4, name='Transformer_OD_Model', **kwargs):
    super().__init__(name=name)
    self.data_loader = data_loader
    self.transformer = transformer
    self.network = WideResidualNetwork(
        input_shape=input_shape, n_classes=n_classes, depth=depth,
        widen_factor=widen_factor, **kwargs)

  def call(self, input_tensor, training=False):
    return self.network(input_tensor, training)

  # TODO: maybe its better to keep keras convention and reduce this to
  #  transformations and leave out data loading
  def fit(self, transform_batch_size, train_batch_size=128, epochs=2,
      **kwargs):
    (x_train, y_train), (x_val, y_val), (
      x_test, y_test) = self.data_loader.get_outlier_detection_datasets()
    self.network.compile(
        general_keys.ADAM, general_keys.CATEGORICAL_CROSSENTROPY,
        [general_keys.ACC])
    x_train_transform, y_train_transform = \
      self.transformer.apply_all_transforms(
          x=x_train, batch_size=transform_batch_size)
    self.network.fit(
        x=x_train_transform, y=tf.keras.utils.to_categorical(y_train_transform),
        batch_size=train_batch_size,
        epochs=epochs, **kwargs)



if __name__ == '__main__':
  print(1)
