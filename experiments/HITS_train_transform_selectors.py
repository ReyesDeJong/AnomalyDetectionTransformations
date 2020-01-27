"""
Selecting best score for large Resnet model, using Simple-Kernel-72 transformers and hits4c-ztf
"""

import itertools
import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from trainers.base_trainer import Trainer
from parameters import loader_keys, general_keys, param_keys
from models.transformer_od import TransformODModel
from modules.geometric_transform import transformations_tf
import tensorflow as tf
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from scripts.transform_selection_clean.training_transform_selection import \
  get_transform_selection_transformer
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
from models.transformer_ensemble_ovo_simple_net_od import \
  EnsembleOVOTransformODSimpleModel


# TRAIN_TIME = 10
# EXP_NAME = 'ZTF_SMALL'


def train_transform_selectors():
  # data loaders
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
  hits_loader = HiTSOutlierLoader(hits_params)
  # ztf_params = {
  #   loader_keys.DATA_PATH: os.path.join(
  #       PROJECT_PATH, '/home/ereyes/Projects/Thesis/datasets/ztf_v1_bogus_added.pkl'),
  #   loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
  #   loader_keys.USED_CHANNELS: [0, 1, 2],
  #   loader_keys.CROP_SIZE: 21,
  #   general_keys.RANDOM_SEED: 42,
  #   loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  # }
  # ztf_loader = ZTFOutlierLoader(ztf_params)
  ztf_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH,
        '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
  }
  ztf_loader = ZTFSmallOutlierLoader(ztf_params)

  # transformers
  transformer_72 = transformations_tf.Transformer()
  plus_kernel_transformer = transformations_tf.PlusKernelTransformer()
  all_kernel_transformer = transformations_tf.KernelTransformer(rotations=True,
                                                                flips=True,
                                                                name='All_Kernel_Transform')

  transformers_list = (
    all_kernel_transformer,
    plus_kernel_transformer,
    transformer_72,
  )
  transformers_list = transformers_list[::-1]
  loaders_list = (hits_loader,)
  loaders_transformer_tuples = list(
      itertools.product(loaders_list, transformers_list))

  for data_loader, transformer in loaders_transformer_tuples:
    (x_train, _), _, _ = data_loader.get_outlier_detection_datasets()
    x_train_shape = x_train.shape[1:]
    del x_train
    print('Initial N Transformations: ', transformer.n_transforms)
    mdl = EnsembleOVOTransformODSimpleModel(
        data_loader=data_loader, transformer=transformer,
        input_shape=x_train_shape)
    transformer = get_transform_selection_transformer(data_loader, mdl,
                                                      transformer)
    del mdl
    print('Final N Transformations: ', transformer.n_transforms)



if __name__ == '__main__':
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  train_transform_selectors()
