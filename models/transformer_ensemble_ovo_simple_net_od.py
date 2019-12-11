"""
Transformer Ensemble OVA
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import tensorflow as tf
from modules.networks.train_step_tf2.deep_hits import DeepHits
from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from parameters import general_keys
from models.transformer_ensemble_ovo_od import EnsembleOVOTransformODModel

"""In situ transformation perform"""


# TODO: create ensemble of models as direct train_step_tf2 model? or no.
#  If so, object from list are by ref, meaning, can they be trained separately?
class EnsembleOVOTransformODSimpleModel(EnsembleOVOTransformODModel):
  def __init__(self, data_loader: ZTFOutlierLoader,
      transformer: AbstractTransformer, input_shape, results_folder_name='',
      name='Ensemble_OVO_Transformer_OD_Simple_Model', **kwargs):
    super().__init__(
        data_loader, transformer, input_shape,
        results_folder_name=results_folder_name, name=name, **kwargs)

  def _get_model_list(self, input_shape, **kwargs):
    models_list = []
    for transform_idx_x in range(self.transformer.n_transforms):
      models_list_x = []
      for transform_idx_y in range(self.transformer.n_transforms):
        if transform_idx_x >= transform_idx_y:
          models_list_x.append(None)
          continue
        network = DeepHits(input_shape, n_classes=2, **kwargs)
        models_list_x.append(network)
      models_list.append(models_list_x)
    return models_list

  # TODO: do a param dict
  def get_network(self, input_shape, n_classes, **kwargs):
    return DeepHits(
        input_shape=input_shape, n_classes=n_classes, **kwargs)


if __name__ == '__main__':
  from parameters import loader_keys
  from modules.geometric_transform import transformations_tf
  from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
  import time
  weights_path = os.path.join(
      PROJECT_PATH,
      'results/transform_selection_1/Ensemble_OVO_Transformer_OD_Simple_Model/Kernel_transformer/Ensemble_OVO_Transformer_OD_Simple_Model_20191211-112609',
      # 'results/transform_selection_2/Ensemble_OVO_Transformer_OD_Simple_Model/72_transformer/Ensemble_OVO_Transformer_OD_Simple_Model_20191210-174727',
      # 'results/transform_selection_3/Ensemble_OVO_Transformer_OD_Simple_Model',
      # '72_transformer/Ensemble_OVO_Transformer_OD_Simple_Model_20191209-192752',
      'checkpoints'
  )

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
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
  data_loader = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader.get_outlier_detection_datasets()
  # transformer = transformations_tf.Transformer()
  transformer = transformations_tf.KernelTransformer(
      flips=True, gauss=False, log=False)
  mdl = EnsembleOVOTransformODSimpleModel(
      data_loader=data_loader, transformer=transformer,
      input_shape=x_train.shape[1:])
  # mdl.build_models()
  mdl.load_model_weights(weights_path)
  train_matrix_scores = mdl.predict_matrix_score(
      x_val, transform_batch_size=1024)
  train_matrix_scores = mdl.predict_matrix_score(
      x_val, transform_batch_size=1024)

  # test_outlier_matrix_scores = mdl.predict_matrix_score(
  #     x_test[y_test==0], transform_batch_size=1024)

  # mdl.fit(x_train, x_val, verbose=0, epochs=1)