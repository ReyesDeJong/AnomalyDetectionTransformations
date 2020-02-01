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
# from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader

TRAIN_TIME = 10
EXP_NAME = 'HITS_NAIVE_TRANSFORMS_D'

def best_score_evaluation(result_folder_name, epochs, patience=0):
  trainer_params = {
    param_keys.RESULTS_FOLDER_NAME: result_folder_name,
    'epochs': epochs,
    'patience': patience,
  }
  # data loaders
  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2, 3],  # [2],#
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
        PROJECT_PATH, '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
  }
  ztf_loader = ZTFSmallOutlierLoader(ztf_params)
  # transformers
  transformer_72 = transformations_tf.Transformer()
  trans_transformer = transformations_tf.TransTransformer()
  kernel_transformer = transformations_tf.KernelTransformer()
  plus_kernel_transformer = transformations_tf.PlusKernelTransformer()
  plus_gauss_transformer = transformations_tf.PlusGaussTransformer()
  plus_laplace_transformer = transformations_tf.PlusLaplaceTransformer()
  # all_kernel_transformer = transformations_tf.KernelTransformer(rotations=True,
  #                                                               flips=True, name='All_Kernel_Transform')
  # trainers
  hits_trainer = Trainer(hits_loader, trainer_params)
  ztf_trainer = Trainer(ztf_loader, trainer_params)

  model_constructors_list = (TransformODModel,)
  transformers_list = (
  # all_kernel_transformer,
  plus_kernel_transformer, plus_gauss_transformer, plus_laplace_transformer,
  transformer_72, kernel_transformer, trans_transformer,

  )
  trainers_list = (hits_trainer,)#(ztf_trainer,)#(ztf_trainer, hits_trainer, )#
  trainer_model_transformer_tuples = list(
      itertools.product(trainers_list, model_constructors_list,
                        transformers_list))

  for trainer, model_constructor, transformer in trainer_model_transformer_tuples:
    trainer.train_model_n_times(model_constructor, transformer,
                                trainer_params, train_times=TRAIN_TIME)

    hits_trainer.create_tables_of_results_folders()


if __name__ == '__main__':
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  best_score_evaluation(
      result_folder_name='%s/resnet_VAL_epochs' % EXP_NAME,
      epochs=1234)

