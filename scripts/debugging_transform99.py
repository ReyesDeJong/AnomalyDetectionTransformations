import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import tensorflow as tf
from modules.networks.train_step_tf2.wide_residual_network import WideResidualNetwork
from modules.geometric_transform import transformations_tf
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from parameters import general_keys
import numpy as np
from modules import dirichlet_utils, utils
from models.transformer_od import TransformODModel
from parameters import loader_keys
import time

if __name__ == '__main__':

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  ztf_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/ztf_v1_bogus_added.pkl'),
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2],
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  ztf_loader = ZTFOutlierLoader(ztf_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = ztf_loader.get_outlier_detection_datasets()
  plus_kernel_transformer = transformations_tf.PlusKernelTransformer()
  model = TransformODModel(
      data_loader=ztf_loader, transformer=plus_kernel_transformer,
      input_shape=x_train.shape[1:])
  weight_path = os.path.join(
      PROJECT_PATH, 'results/best_scores_refact/best_scores_resnet_1_epochs/Transformer_OD_Model/PlusKernel_Transformer/Transformer_OD_Model_20191223-193141/checkpoints/final_weights.ckpt')
  model.load_weights(weight_path)
  met_dict = model.evaluate_od(
      x_train, x_test, y_test, ztf_loader.name, general_keys.REAL, x_val)