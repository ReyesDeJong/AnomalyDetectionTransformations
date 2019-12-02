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
from modules import score_functions

"""In situ transformation perform"""


# TODO: create ensemble of models as direct keras model? or no.
#  If so, object from list are by ref, meaning, can they be trained separately?
class EnsembleOVATransformODModel(TransformODModel):
  def __init__(self, data_loader: ZTFOutlierLoader,
      transformer: AbstractTransformer, input_shape, depth=10,
      widen_factor=4, results_folder_name='',
      name='Ensemble_OVA_Transformer_OD_Model', **kwargs):
    super(TransformODModel, self).__init__(name=name)
    self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    self.main_model_path = self.create_main_model_paths(results_folder_name,
                                                        self.name)
    utils.check_paths(self.main_model_path)
    self.data_loader = data_loader
    self.transformer = transformer
    self.models_list = self._get_model_list(input_shape, depth, widen_factor,
                                            **kwargs)

  def call(self, input_tensor, training=False):
    output = []
    for model in self.models_list:
      output.append(model(input_tensor, training)[:, 1])
    return tf.stack(output, axis=-1)

  def _get_model_list(self, input_shape, depth, widen_factor, **kwargs):
    models_list = []
    for transform_idx in range(self.transformer.n_transforms):
      network = WideResidualNetwork(
          input_shape=input_shape, n_classes=2,
          depth=depth, widen_factor=widen_factor, **kwargs)
      models_list.append(network)
    return models_list

  # TODO: make this an external utils function
  def replicate_to_size(self, data_array, size):
    if len(data_array) < size:
      return self.replicate_to_size(np.concatenate([data_array, data_array]),
                                    size)
    else:
      size_left = size - len(data_array)
      return np.concatenate([data_array, data_array[:size_left]])

  def fit(self, x_train, x_val, transform_batch_size=512, train_batch_size=128,
      epochs=2, **kwargs):
    self.create_specific_model_paths()
    x_train_transform, y_train_transform = \
      self.transformer.apply_all_transforms(
          x=x_train, batch_size=transform_batch_size)
    x_val_transform, y_val_transform = \
      self.transformer.apply_all_transforms(
          x=x_val, batch_size=transform_batch_size)
    for t_ind, model in enumerate(self.models_list):
      model.compile(general_keys.ADAM, general_keys.CATEGORICAL_CROSSENTROPY,
                    [general_keys.ACC])
      # TODO: make this step a function

      # separate inliers as an specific transform an the rest as outlier for an specific classifier, balance by replication
      selected_transformation_to_train = t_ind
      selected_transform_indxs_train = \
        np.where(y_train_transform == selected_transformation_to_train)[0]
      non_transform_indxs_train = \
        np.where(y_train_transform != selected_transformation_to_train)[0]
      selected_transform_indxs_val = \
        np.where(y_val_transform == selected_transformation_to_train)[0]
      non_transform_indxs_val = \
        np.where(y_val_transform != selected_transformation_to_train)[0]
      oversampled_selected_trans_idxs_train = self.replicate_to_size(
          selected_transform_indxs_train, len(non_transform_indxs_train))
      subsamples_val_idxs = np.random.choice(
          non_transform_indxs_val, len(selected_transform_indxs_val),
          replace=False)
      train_x_binary = np.concatenate(
          [x_train_transform[oversampled_selected_trans_idxs_train],
           x_train_transform[non_transform_indxs_train]])
      train_y_binary = np.concatenate(
          [np.ones_like(oversampled_selected_trans_idxs_train),
           np.zeros_like(non_transform_indxs_train)])
      val_x_binary = np.concatenate(
          [x_val_transform[selected_transform_indxs_val],
           x_val_transform[subsamples_val_idxs]])
      val_y_binary = np.concatenate([np.ones_like(selected_transform_indxs_val),
                                     np.zeros_like(subsamples_val_idxs)])
      print('Training Model %i' % t_ind)
      print('Train_size: ', np.unique(train_y_binary, return_counts=True))
      print('Val_size: ', np.unique(val_y_binary, return_counts=True))

      es = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss', mode='min', verbose=1, patience=0,
          restore_best_weights=True)
      model.fit(
          x=train_x_binary,
          y=tf.keras.utils.to_categorical(train_y_binary),
          validation_data=(
            val_x_binary, tf.keras.utils.to_categorical(val_y_binary)),
          batch_size=train_batch_size,
          epochs=epochs, callbacks=[es], **kwargs)
      weight_path = os.path.join(self.checkpoint_folder,
                                 'final_weights_model%i.h5' % t_ind)
      # TODO: what happens if y do self.save_weights??
      model.save_weights(weight_path)

  def predict_matrix_score(self, x, transform_batch_size=512,
      predict_batch_size=1024, **kwargs):
    n_transforms = self.transformer.n_transforms
    x_transformed, y_transformed = self.transformer.apply_all_transforms(
        x, transform_batch_size)
    matrix_scores = np.zeros((len(x), n_transforms, n_transforms))
    # TODO: paralelice this
    for model_t_ind in tqdm(range(n_transforms)):
      x_pred_model_t_ind = self.models_list[model_t_ind].predict(x_transformed,
                                                                 batch_size=predict_batch_size)
      for t_ind in range(n_transforms):
        ind_x_pred_model_t_ind_queal_to_t_ind = \
          np.where(y_transformed == t_ind)[0]
        matrix_scores[:, model_t_ind, t_ind] += x_pred_model_t_ind[
                                                  ind_x_pred_model_t_ind_queal_to_t_ind][
                                                :, 1]
    del x_transformed, y_transformed
    return matrix_scores

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
    diri_scores_transform = np.zeros(len(x_eval))
    for t_ind in range(n_transforms):
      observed_dirichlet = utils.normalize_sum1(
          matrix_scores_train[:, :, t_ind])
      x_eval_p = utils.normalize_sum1(matrix_scores_eval[:, :, t_ind])
      diri_scores_transform += dirichlet_utils.dirichlet_score(
          observed_dirichlet, x_eval_p)
    diri_scores_transform /= n_transforms
    diri_scores_model = np.zeros(len(x_eval))
    for mdl_ind in range(n_transforms):
      observed_dirichlet = utils.normalize_sum1(matrix_scores_train[:, mdl_ind, :])
      x_eval_p = utils.normalize_sum1(matrix_scores_eval[:, mdl_ind, :])
      diri_scores_model += dirichlet_utils.dirichlet_score(
          observed_dirichlet, x_eval_p)
    diri_scores_model /= n_transforms
    return matrix_scores_eval, diri_scores_transform, diri_scores_model

  def get_scores_dict(self, x_train, x_eval,
      transform_batch_size=512, predict_batch_size=1024, **kwargs):
    matrix_scores, diri_scores_trans, diri_scores_mdl = \
      self.predict_matrix_and_dirichlet_score(
          x_train, x_eval, transform_batch_size, predict_batch_size, **kwargs)
    matrix_scores = matrix_scores / self.transformer.n_transforms
    scores_dict = {
      general_keys.DIRI_OVA_MDL: diri_scores_mdl,
      general_keys.DIRI_OVA_TRANS: diri_scores_trans,
      general_keys.MATRIX_TRACE: np.trace(matrix_scores, axis1=1, axis2=2),
      general_keys.ENTROPY: -1 * score_functions.get_entropy(matrix_scores),
      general_keys.CROSS_ENTROPY: -1 * score_functions.get_xH(self.transformer,
                                                              matrix_scores)
    }
    return scores_dict


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
  model = EnsembleOVATransformODModel(
      data_loader=ztf_od_loader, transformer=transformer,
      input_shape=x_train.shape[1:])

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
