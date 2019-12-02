"""
Transformer Ensemble OVA
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
import numpy as np
from modules import utils, dirichlet_utils
import datetime
from tqdm import tqdm
from models.transformer_od import TransformODModel

"""In situ transformation perform"""


# TODO: create ensemble of models as direct keras model? or no.
#  If so, object from list are by ref, meaning, can they be trained separately?
class EnsembleOVOTransformODModel(TransformODModel):
  def __init__(self, data_loader: ZTFOutlierLoader,
      transformer: AbstractTransformer, input_shape, depth=10,
      widen_factor=4, results_folder_name='',
      name='Ensemble_OVO_Transformer_OD_Model', **kwargs):
    super(TransformODModel, self).__init__(name=name)
    self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    self.main_model_path = self.create_main_model_paths(results_folder_name,
                                                        self.name)
    utils.check_paths(self.main_model_path)
    self.data_loader = data_loader
    self.transformer = transformer
    self.models_list = self._get_model_list(input_shape, depth, widen_factor,
                                            **kwargs)

  # def call(self, input_tensor, training=False):
  #   output = []
  #   for model_list_x in self.models_list:
  #     output_x = []
  #     for model in model_list_x:
  #       output_x.append(model(input_tensor, training)[:, 1])
  #     output_x = tf.stack(output_x, axis=-1)
  #     output.append(output_x)
  #   return tf.stack(output, axis=-1)

  def _get_model_list(self, input_shape, depth, widen_factor, **kwargs):
    models_list = []
    for transform_idx_x in range(self.transformer.n_transforms):
      models_list_x = []
      for transform_idx_y in range(self.transformer.n_transforms):
        if transform_idx_x >= transform_idx_y:
          models_list_x.append(None)
          continue
        network = WideResidualNetwork(
            input_shape=input_shape, n_classes=2,
            depth=depth, widen_factor=widen_factor, **kwargs)
        models_list_x.append(network)
      models_list.append(models_list_x)
    return models_list

  def fit(self, x_train, x_val, transform_batch_size=512, train_batch_size=128,
      epochs=2, **kwargs):
    self.create_specific_model_paths()
    x_train_transform, y_train_transform = \
      self.transformer.apply_all_transforms(
          x=x_train, batch_size=transform_batch_size)
    x_val_transform, y_val_transform = \
      self.transformer.apply_all_transforms(
          x=x_val, batch_size=transform_batch_size)
    for model_ind_x in range(self.transformer.n_transforms):
      for t_mdl_ind_y, model_y in enumerate(self.models_list[model_ind_x]):
        if model_ind_x >= t_mdl_ind_y:
          continue
        model_y.compile(
            general_keys.ADAM, general_keys.CATEGORICAL_CROSSENTROPY,
            [general_keys.ACC])
        # TODO: make this step a function

        # separate inliers as an specific transform an the rest as outlier for an specific classifier, balance by replication
        transformation_x_to_train = model_ind_x
        # this to be class 1
        transformation_y_to_train = t_mdl_ind_y
        transform_x_indxs_train = \
          np.where(y_train_transform == transformation_x_to_train)[0]
        transform_y_indxs_train = \
          np.where(y_train_transform == transformation_y_to_train)[0]
        transform_x_indxs_val = \
          np.where(y_val_transform == transformation_x_to_train)[0]
        transform_y_indxs_val = \
          np.where(y_val_transform == transformation_y_to_train)[0]
        train_x_binary = np.concatenate(
            [x_train_transform[transform_x_indxs_train],
             x_train_transform[transform_y_indxs_train]])
        train_y_binary = np.concatenate(
            [np.zeros_like(transform_x_indxs_train),
             np.ones_like(transform_y_indxs_train)])
        val_x_binary = np.concatenate(
            [x_val_transform[transform_x_indxs_val],
             x_val_transform[transform_y_indxs_val]])
        val_y_binary = np.concatenate([np.zeros_like(transform_x_indxs_val),
                                       np.ones_like(transform_y_indxs_val)])
        print('Training Model x%i (0) y%i (1)' % (model_ind_x, t_mdl_ind_y))
        print('Train_size: ', np.unique(train_y_binary, return_counts=True))
        print('Val_size: ', np.unique(val_y_binary, return_counts=True))

        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', verbose=1, patience=0,
            restore_best_weights=True)
        model_y.fit(
            x=train_x_binary,
            y=tf.keras.utils.to_categorical(train_y_binary),
            validation_data=(
              val_x_binary, tf.keras.utils.to_categorical(val_y_binary)),
            batch_size=train_batch_size,
            epochs=epochs, callbacks=[es], **kwargs)
        weight_path = os.path.join(self.checkpoint_folder,
                                   'final_weights_modelx%iy%i.h5' % (
                                     model_ind_x, t_mdl_ind_y))
        # TODO: what happens if y do self.save_weights??
        model_y.save_weights(weight_path)

  def predict_matrix_score(self, x, transform_batch_size=512,
      predict_batch_size=1024, **kwargs):
    n_transforms = self.transformer.n_transforms
    x_transformed, y_transformed = self.transformer.apply_all_transforms(
        x, transform_batch_size)
    matrix_scores = np.zeros((len(x), n_transforms, n_transforms))
    for model_t_x in tqdm(range(n_transforms)):
      for t_mdl_ind_y in range(n_transforms):
        if model_t_x >= t_mdl_ind_y:
          continue
        ind_x_pred_model_t_x_queal_to_t_ind = \
          np.where(y_transformed == t_mdl_ind_y)[0]
        x_pred_model_x_t_ind = self.models_list[model_t_x][t_mdl_ind_y].predict(
            x_transformed[ind_x_pred_model_t_x_queal_to_t_ind],
            batch_size=predict_batch_size)
        matrix_scores[:, model_t_x, t_mdl_ind_y] += x_pred_model_x_t_ind[:, 0]
    del x_transformed, y_transformed
    return self._post_process_matrix_score(matrix_scores)

  def _post_process_matrix_score(self, matrix_score):
    """fill diagonal with 1- mean of row, and triangle bottom with reflex of
    up"""
    for i_x in range(matrix_score.shape[1]):
      for i_y in range(matrix_score.shape[2]):
        if i_x == i_y:
          matrix_score[:, i_x, i_y] = 1 - np.mean(matrix_score[:, i_x, :],
                                                  axis=-1)
        elif i_x > i_y:
          matrix_score[:, i_x, i_y] = matrix_score[:, i_y, i_x]
    return matrix_score

  # TODO: Dunno how to proceed with dirichlet in this case,
  #  can flatten vector, or just us diagonal
  def predict_matrix_and_dirichlet_score(self, x_train, x_eval,
      transform_batch_size=512, predict_batch_size=1024,
      **kwargs):
    n_transforms = self.transformer.n_transforms
    matrix_scores_train = self.predict_matrix_score(
        x_train, transform_batch_size, predict_batch_size, **kwargs)
    matrix_scores_eval = self.predict_matrix_score(
        x_eval, transform_batch_size, predict_batch_size, **kwargs)
    # this across transforms considerates all models, and their discriminative power
    # acros models is ingle model discriminative power
    # TODO: implement both
    diri_scores = np.zeros(len(x_eval))
    for t_ind in range(n_transforms):
      observed_dirichlet = utils.normalize_sum1(matrix_scores_train[:, :, t_ind])
      x_eval_p = utils.normalize_sum1(matrix_scores_eval[:, :, t_ind])
      diri_scores += dirichlet_utils.dirichlet_score(
          observed_dirichlet, x_eval_p)
    diri_scores /= n_transforms
    return matrix_scores_eval, diri_scores




if __name__ == '__main__':
  from parameters import loader_keys
  from modules.geometric_transform.transformations_tf import TransTransformer
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
  transformer = TransTransformer()
  model = EnsembleOVOTransformODModel(
      data_loader=ztf_od_loader, transformer=transformer,
      input_shape=x_train.shape[1:])
  # weight_path = os.path.join(PROJECT_PATH, 'results', model.name,
  #                            'my_checkpoint.h5')
  # model.load_weights(weight_path)
  model.fit(x_train, x_val)
  start_time = time.time()
  model.create_specific_model_paths()
  met_dict = model.evaluate_od(
      x_train, x_test, y_test, 'ztf-real-bog-v1', 'real', x_val,
      save_hist_folder_path=model.specific_model_folder)
  print(
      "Time model.evaluate_od %s" % utils.timer(
          start_time, time.time()),
      flush=True)

  # start_time = time.time()
  # pred = model.network.predict(x_test, batch_size=1024)
  # print("Time model.pred %s" % utils.timer(start_time, time.time()), flush=True)
  # print(pred.shape)
  #
  # start_time = time.time()
  # pred = model.predict_dirichlet_score(x_train, x_test)
  # print("Time model.predict_dirichlet_score %s" % utils.timer(start_time,
  #                                                             time.time()),
  #       flush=True)
  # print(pred.shape)
  #
  # start_time = time.time()
  # pred = model.predict_matrix_score(x_test)
  # print("Time model.predict_matrix_score %s" % utils.timer(start_time,
  #                                                          time.time()),
  #       flush=True)
  # print(pred.shape)
  #
  # start_time = time.time()
  # pred_mat, pred_score = model.predict_matrix_and_dirichlet_score(x_train, x_test)
  # print(
  #     "Time model.predict_matrix_and_dirichlet_score %s" % utils.timer(
  #         start_time, time.time()),
  #     flush=True)
  # print(pred_mat.shape, pred_score.shape)
  """
  Time model.pred 00:00:04.92
  (4302, 72)
  Time model.predict_dirichlet_score 00:01:13.92
  (4302,)
  Appliying all transforms to set of shape (4302, 21, 21, 3)
  Time model.predict_matrix_score 00:00:08.38
  (4302, 72, 72)
  Time model.predict_matrix_and_dirichlet_score 00:01:14.36
  """

  # start_time = time.time()
  # dict = model.get_scores_dict(x_train, x_test)
  # print(
  #     "Time model.get_scores_dict %s" % utils.timer(
  #         start_time, time.time()),
  #     flush=True)

  # #
  # # # pprint.pprint(met_dict)
  # print('\nroc_auc')
  # for key in met_dict.keys():
  #   print(key, met_dict[key]['roc_auc'])
  # print('\nacc_at_percentil')
  # for key in met_dict.keys():
  #   print(key, met_dict[key]['acc_at_percentil'])
  # print('\nmax_accuracy')
  # for key in met_dict.keys():
  #   print(key, met_dict[key]['max_accuracy'])
  #
  #

  # model.save_weights(
  #   os.path.join(PROJECT_PATH, 'results', model.name, 'my_checkpoint.h5'))
