"""
geometric trasnformer for outlier detection in tf2, simpler network
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import tensorflow as tf
# from modules.networks.deep_hits import DeepHits
from modules.networks.train_step_tf2.deep_hits import DeepHits
from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from parameters import general_keys
from modules import utils
import datetime
from models.transformer_od import TransformODModel
from modules.print_manager import PrintManager

"""In situ transformation perform"""


class TransformODSimpleModel(TransformODModel):
  def __init__(self, data_loader: ZTFOutlierLoader,
      transformer: AbstractTransformer, input_shape, results_folder_name='',
      name='Transformer_OD_Simple_Model',
      **kwargs):
    super(TransformODModel, self).__init__(name=name)
    self.print_manager = PrintManager()
    # self._init_gpu_usage()
    self.data_loader = data_loader
    self.transformer = transformer
    self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    self.main_model_path = self.create_main_model_paths(results_folder_name,
                                                        self.name)
    self.create_specific_model_paths()
    utils.check_paths(self.main_model_path)
    self.network = self.get_network(
        input_shape=input_shape, n_classes=self.transformer.n_transforms,
        model_path=self.specific_model_folder, **kwargs)
    self.percentile = 97.73  # 95.45

  # TODO: do a param dict
  def get_network(self, input_shape, n_classes, model_path, **kwargs):
    return DeepHits(
        input_shape=input_shape, n_classes=n_classes, model_path=model_path,
        **kwargs)


if __name__ == '__main__':
  from parameters import loader_keys
  from modules.geometric_transform.transformations_tf import Transformer
  import time

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  data_loader_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/ztf_v1_bogus_added.pkl'),
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2],
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  ztf_od_loader = ZTFOutlierLoader(data_loader_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = ztf_od_loader.get_outlier_detection_datasets()
  transformer = Transformer()
  model = TransformODSimpleModel(
      data_loader=ztf_od_loader, transformer=transformer,
      input_shape=x_train.shape[1:])
  model.build(tuple([None] + list(x_train.shape[1:])))
  # print(model.network.model().summary())
  weight_path = os.path.join(PROJECT_PATH, 'results', model.name,
                             'my_checkpoint_simple.h5')
  if os.path.exists(weight_path):
    model.load_weights(weight_path).expect_partial()
  else:
    model.fit(x_train, x_val)

  start_time = time.time()
  met_dict = model.evaluate_od(
      x_train, x_test, y_test, 'ztf-real-bog-v1', 'real', x_val)
  print(
      "Time model.evaluate_od %s" % utils.timer(
          start_time, time.time()),
      flush=True)

  print('\nroc_auc')
  for key in met_dict.keys():
    print(key, met_dict[key]['roc_auc'])
  print('\nacc_at_percentil')
  for key in met_dict.keys():
    print(key, met_dict[key]['acc_at_percentil'])
  print('\nmax_accuracy')
  for key in met_dict.keys():
    print(key, met_dict[key]['max_accuracy'])

  model.save_weights(weight_path)
