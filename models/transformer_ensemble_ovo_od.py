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
from tensorflow.keras.utils import to_categorical
from models.transformer_od import TransformODModel
import itertools
import matplotlib.pyplot as plt

"""In situ transformation perform"""


# TODO: create ensemble of models as direct train_step_tf2 model? or no.
#  If so, object from list are by ref, meaning, can they be trained separately?
class EnsembleOVOTransformODModel(TransformODModel):
  def __init__(self, data_loader: ZTFOutlierLoader,
      transformer: AbstractTransformer, input_shape, depth=10,
      widen_factor=4, results_folder_name='',
      name='Ensemble_OVO_Transformer_OD_Model', build=True, verbose=True,
      **kwargs):
    super(TransformODModel, self).__init__(name=name)
    self.verbose = verbose
    self.builder_input_shape = input_shape
    self.depth = 10
    self.widen_factor = widen_factor
    self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    self.main_model_path = self.create_main_model_paths(results_folder_name,
                                                        self.name)
    utils.check_paths(self.main_model_path)
    self.data_loader = data_loader
    self.transformer = transformer
    self.models_index_tuples = self._get_models_index_tuples()
    if build:
      self.models_list = self._get_model_list(input_shape, depth=depth,
                                              widen_factor=widen_factor,
                                              **kwargs)
    self.create_commonmodel_paths()

  def create_commonmodel_paths(self):
    self.common_to_all_models_transform_selection_folder = os.path.join(
        self.main_model_path, 'transform_selection', self.data_loader.name,
        self.transformer.name)
    self.common_to_all_models_transform_selection_checkpoints_folder = os.path.join(
        self.common_to_all_models_transform_selection_folder,
        'checkpoints')
    self.common_to_all_models_transform_selection_results_folder = os.path.join(
        self.common_to_all_models_transform_selection_folder,
        'results')
    # self.tb_path = os.path.join(self.model_path, 'tb_summaries')
    utils.check_paths(
        [self.common_to_all_models_transform_selection_folder,
         self.common_to_all_models_transform_selection_checkpoints_folder,
         self.common_to_all_models_transform_selection_results_folder])

  def _get_models_index_tuples(self):
    transforms_arange = np.arange(self.transformer.n_transforms)
    transforms_tuples = list(
        itertools.product(transforms_arange, transforms_arange))
    models_index_tuples = []
    for x_y_tuple in transforms_tuples:
      if x_y_tuple[0] < x_y_tuple[1]:
        models_index_tuples.append(x_y_tuple)
    return models_index_tuples

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
    for transform_idx_x in tqdm(range(self.transformer.n_transforms)):
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

  def compile_and_build_models(self):
    for x_y_tuple in tqdm(self.models_index_tuples):
      model_ind_x = x_y_tuple[0]
      t_mdl_ind_y = x_y_tuple[1]
      model_y = self.models_list[model_ind_x][t_mdl_ind_y]
      model_y.build((None,) + self.builder_input_shape)
      model_y.compile(
          general_keys.ADAM, general_keys.CATEGORICAL_CROSSENTROPY,
          [general_keys.ACC])

  def build_models(self):
    print('\nBuilding Models\n')
    for x_y_tuple in tqdm(self.models_index_tuples):
      model_ind_x = x_y_tuple[0]
      t_mdl_ind_y = x_y_tuple[1]
      model_y = self.models_list[model_ind_x][t_mdl_ind_y]
      model_y.build((None,) + self.builder_input_shape)

  def _get_binary_data(self, data_transformed, labels_transformed, x_ind,
      y_ind):
    transform_x_indxs = np.where(labels_transformed == x_ind)[0]
    transform_y_indxs = np.where(labels_transformed == y_ind)[0]
    data_binary = np.concatenate(
        [data_transformed[transform_x_indxs],
         data_transformed[transform_y_indxs]])
    labels_binary = np.concatenate(
        [np.zeros_like(transform_x_indxs), np.ones_like(transform_y_indxs)])
    return data_binary, labels_binary

  def _large_model_fit(self, x_train, x_val, transform_batch_size,
      train_batch_size, epochs, save_weights_in_common_path, **kwargs):
    print('Fit')
    x_train_transform, y_train_transform = \
      self.transformer.apply_all_transforms(
          x=x_train, batch_size=transform_batch_size)
    x_val_transform, y_val_transform = \
      self.transformer.apply_all_transforms(
          x=x_val, batch_size=transform_batch_size)
    for x_y_tuple in tqdm(self.models_index_tuples):
      model_ind_x = x_y_tuple[0]
      t_mdl_ind_y = x_y_tuple[1]
      model = self.get_network(
          input_shape=self.builder_input_shape, n_classes=2, depth=self.depth,
          widen_factor=self.widen_factor)
      # if model_ind_x >= t_mdl_ind_y:
      #   raise ValueError('Condition not met!')
      #   continue
      # model.compile(
      #     general_keys.ADAM, general_keys.CATEGORICAL_CROSSENTROPY,
      #     [general_keys.ACC])
      # separate inliers as an specific transform an the rest as outlier for an specific classifier, balance by replication
      transform_x_ind_to_train = model_ind_x
      # this to be class 1
      transform_y_ind_to_train = t_mdl_ind_y
      train_x_binary, train_y_binary = self._get_binary_data(
          x_train_transform, y_train_transform, transform_x_ind_to_train,
          transform_y_ind_to_train)
      val_x_binary, val_y_binary = self._get_binary_data(
          x_val_transform, y_val_transform, transform_x_ind_to_train,
          transform_y_ind_to_train)
      # print('Training Model x%i (0) y%i (1)' % (model_ind_x, t_mdl_ind_y))
      # print('Train_size: ', np.unique(train_y_binary, return_counts=True))
      # print('Val_size: ', np.unique(val_y_binary, return_counts=True))
      es = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss', mode='min', patience=0,
          restore_best_weights=True, **kwargs)
      callbacks = [es]
      validation_data = (
        val_x_binary, tf.keras.utils.to_categorical(val_y_binary))
      if epochs < 3:
        callbacks = None
        validation_data = None
      model.fit(
          x=train_x_binary,
          y=tf.keras.utils.to_categorical(train_y_binary),
          validation_data=validation_data,
          batch_size=train_batch_size,
          epochs=epochs, callbacks=callbacks, **kwargs)
      weights_name = 'final_weights_modelx%iy%i.ckpt' % (
        model_ind_x, t_mdl_ind_y)
      weight_path = os.path.join(self.checkpoint_folder, weights_name)
      common_to_all_models_weight_path = os.path.join(
          self.common_to_all_models_transform_selection_checkpoints_folder,
          weights_name)
      # print(os.path.abspath(weight_path))
      # TODO: what happens if y do self.save_weights??
      model.save_weights(weight_path)
      if save_weights_in_common_path:
        model.save_weights(common_to_all_models_weight_path)
      del model, validation_data, val_x_binary, val_y_binary, train_y_binary, train_x_binary
    self.load_model_weights(self.checkpoint_folder)

  def fit(self, x_train, x_val, transform_batch_size=512, train_batch_size=128,
      epochs=2, save_weights_in_common_path=True, **kwargs):
    """Large fit always happens"""
    self.create_specific_model_paths()
    if len(self.models_index_tuples) > 0:
      return self._large_model_fit(
          x_train, x_val, transform_batch_size, train_batch_size, epochs,
          save_weights_in_common_path,
          **kwargs)
    # # transforming data
    # x_train_transform, y_train_transform = \
    #   self.transformer.apply_all_transforms(
    #       x=x_train, batch_size=transform_batch_size)
    # x_val_transform, y_val_transform = \
    #   self.transformer.apply_all_transforms(
    #       x=x_val, batch_size=transform_batch_size)
    # for x_y_tuple in tqdm(self.models_index_tuples):
    #   model_ind_x = x_y_tuple[0]
    #   t_mdl_ind_y = x_y_tuple[1]
    #   model_y = self.models_list[model_ind_x][t_mdl_ind_y]
    #   # if model_ind_x >= t_mdl_ind_y:
    #   #   raise ValueError('Condition not met!')
    #   #   continue
    #   # model_y.compile(
    #   #     general_keys.ADAM, general_keys.CATEGORICAL_CROSSENTROPY,
    #   #     [general_keys.ACC])
    #   # separate inliers as an specific transform an the rest as outlier for an specific classifier, balance by replication
    #   transform_x_ind_to_train = model_ind_x
    #   # this to be class 1
    #   transform_y_ind_to_train = t_mdl_ind_y
    #   train_x_binary, train_y_binary = self._get_binary_data(
    #       x_train_transform, y_train_transform, transform_x_ind_to_train,
    #       transform_y_ind_to_train)
    #   val_x_binary, val_y_binary = self._get_binary_data(
    #       x_val_transform, y_val_transform, transform_x_ind_to_train,
    #       transform_y_ind_to_train)
    #   # print('Training Model x%i (0) y%i (1)' % (model_ind_x, t_mdl_ind_y))
    #   # print('Train_size: ', np.unique(train_y_binary, return_counts=True))
    #   # print('Val_size: ', np.unique(val_y_binary, return_counts=True))
    #   es = tf.keras.callbacks.EarlyStopping(
    #       monitor='val_loss', mode='min', patience=0,
    #       restore_best_weights=True, **kwargs)
    #   callbacks = [es]
    #   validation_data = (
    #     val_x_binary, tf.keras.utils.to_categorical(val_y_binary))
    #   if epochs < 3:
    #     callbacks = None
    #     validation_data = None
    #   model_y.fit_tf(
    #       x=train_x_binary,
    #       y=tf.keras.utils.to_categorical(train_y_binary),
    #       validation_data=validation_data,
    #       batch_size=train_batch_size,
    #       epochs=epochs, callbacks=callbacks, **kwargs)
    #   weight_path = os.path.join(self.checkpoint_folder,
    #                              'final_weights_modelx%iy%i.ckpt' % (
    #                                model_ind_x, t_mdl_ind_y))
    #
    #   # TODO: what happens if y do self.save_weights??
    #   model_y.save_weights(weight_path)
    #   # del model_y, validation_data, val_x_binary, val_y_binary, train_y_binary, train_x_binary

  # TODO: make it not reflected, because data is not the same,
  #  some transformation comparisons are being left out
  def predict_matrix_score(self, x, transform_batch_size=512,
      predict_batch_size=1024, **kwargs):
    print('\nPredicting Matrix Score\n')
    n_transforms = self.transformer.n_transforms
    x_transformed, y_transformed = self.transformer.apply_all_transforms(
        x, transform_batch_size)
    matrix_scores = np.zeros((len(x), n_transforms, n_transforms))
    for x_y_tuple in tqdm(self.models_index_tuples):
      model_t_x = x_y_tuple[0]
      t_mdl_ind_y = x_y_tuple[1]
      ind_x_pred_model_t_x_queal_to_t_ind = \
        np.where(y_transformed == t_mdl_ind_y)[0]
      # x_pred_model_x_t_ind = self.models_list[model_t_x][
      #   t_mdl_ind_y].predict_tf(
      #     x_transformed[ind_x_pred_model_t_x_queal_to_t_ind],
      #     batch_size=predict_batch_size, **kwargs)
      x_pred_model_x_t_ind = self.models_list[model_t_x][
        t_mdl_ind_y](
          x_transformed[ind_x_pred_model_t_x_queal_to_t_ind])
      matrix_scores[:, model_t_x, t_mdl_ind_y] += x_pred_model_x_t_ind[:, 0]
    # del x_transformed, y_transformed
    return self._post_process_matrix_score(matrix_scores)

  def get_acc_matrix(self, x, transform_batch_size=512,
      predict_batch_size=1024, **kwargs):
    print('\nPredicting Acc Matrix\n')
    n_transforms = self.transformer.n_transforms
    x_transformed, y_transformed = self.transformer.apply_all_transforms(
        x, transform_batch_size)
    matrix_scores = np.zeros((n_transforms, n_transforms))
    for x_y_tuple in tqdm(self.models_index_tuples):
      model_t_x = x_y_tuple[0]
      t_mdl_ind_y = x_y_tuple[1]
      x_binary, y_binary = self._get_binary_data(
          x_transformed, y_transformed, model_t_x, t_mdl_ind_y)
      acc_model_x_t_ind = self.models_list[model_t_x][
        t_mdl_ind_y].eval_tf(
          x_binary, to_categorical(y_binary), predict_batch_size)[
        general_keys.ACCURACY]
      matrix_scores[model_t_x, t_mdl_ind_y] += acc_model_x_t_ind
    # del x_transformed, y_transformed
    return self._post_process_acc_matrix(matrix_scores)

  def _post_process_matrix_score(self, matrix_score):
    """fill diagonal with 1- mean of row, and triangle bottom with reflex of
    up"""
    for i_x in range(matrix_score.shape[-2]):
      for i_y in range(matrix_score.shape[-1]):
        if i_x == i_y:
          matrix_score[:, i_x, i_y] = 1 - np.mean(matrix_score[:, i_x, :],
                                                  axis=-1)
        elif i_x > i_y:
          matrix_score[:, i_x, i_y] = matrix_score[:, i_y, i_x]
    return matrix_score

  def _post_process_acc_matrix(self, matrix_score):
    """fill diagonal with 1- mean of row, and triangle bottom with reflex of
    up"""
    for i_x in range(matrix_score.shape[-2]):
      for i_y in range(matrix_score.shape[-1]):
        if i_x == i_y:
          matrix_score[i_x, i_y] = 1 - np.mean(matrix_score[i_x, :],
                                               axis=-1)
        elif i_x > i_y:
          matrix_score[i_x, i_y] = matrix_score[i_y, i_x]
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
      observed_dirichlet = utils.normalize_sum1(
          matrix_scores_train[:, :, t_ind])
      x_eval_p = utils.normalize_sum1(matrix_scores_eval[:, :, t_ind])
      diri_scores += dirichlet_utils.dirichlet_score(
          observed_dirichlet, x_eval_p)
    diri_scores /= n_transforms
    return matrix_scores_eval, diri_scores

  def load_model_weights(self, checkpoints_folder):
    print('\nLoading Models\n')
    for x_y_tuple in tqdm(self.models_index_tuples):
      model_ind_x = x_y_tuple[0]
      t_mdl_ind_y = x_y_tuple[1]
      weights_name = 'final_weights_modelx%iy%i.ckpt' % (
        model_ind_x, t_mdl_ind_y)
      weights_path = os.path.join(checkpoints_folder, weights_name)
      self.models_list[model_ind_x][t_mdl_ind_y].load_weights(
          weights_path).expect_partial()

  def plot_score_acc_matrices(self, matrices, N_to_plot=1):
    if len(matrices.shape) == 2:
      matrices = matrices[None]
    for i in range(N_to_plot):
      index = np.random.randint(len(matrices))
      plt.imshow(matrices[index])
      plt.show()


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
