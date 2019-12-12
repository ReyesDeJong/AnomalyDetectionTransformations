"""
Model that recives as input transformed data
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import tensorflow as tf
from modules.geometric_transform.transformations_tf import AbstractTransformer
from parameters import general_keys
import numpy as np
from modules import dirichlet_utils
from modules import score_functions
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from models.transformer_od import TransformODModel

"""In situ transformation perform"""


def _train_ocsvm_and_score(params, x_train, val_labels, x_val):
  return np.mean(val_labels ==
                 OneClassSVM(**params).fit(x_train).predict(
                     x_val))


# TODO: think if its better to create a trainer instead of an encapsulated model
class AlreadyTransformODModel(TransformODModel):
  def __init__(self, transformer: AbstractTransformer, input_shape, depth=10,
      widen_factor=4, results_folder_name='',
      name='Already_Transformer_OD_Model', **kwargs):
    super().__init__(None,
                     transformer, input_shape, depth,
                     widen_factor, results_folder_name,
                     name, **kwargs)

  def fit(self, train_data, val_data, transform_batch_size=512,
      train_batch_size=128,
      epochs=2, **kwargs):
    self.create_specific_model_paths()
    # ToDo: must be network? or just self.compile???
    self.network.compile(
        general_keys.ADAM, general_keys.CATEGORICAL_CROSSENTROPY,
        [general_keys.ACC])
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=0,
        restore_best_weights=True)
    self.network.fit(
        x=train_data[0], y=tf.keras.utils.to_categorical(train_data[1]),
        validation_data=(
          val_data[0], tf.keras.utils.to_categorical(val_data[1])),
        batch_size=train_batch_size,
        epochs=epochs, callbacks=[es], **kwargs)
    weight_path = os.path.join(self.checkpoint_folder,
                               'final_weights.h5')
    self.save_weights(weight_path)
    # del train_data, val_data

  def predict_dirichlet_score(self, train_data, eval_data,
      transform_batch_size=512, predict_batch_size=1024,
      **kwargs):
    _, diri_scores = self.predict_matrix_and_dirichlet_score(
        train_data, eval_data, transform_batch_size, predict_batch_size,
        **kwargs)
    return diri_scores

  # TODO: implement pre-transofrmed, in-situ-all, efficient transforming
  def predict_matrix_score(self, x_data, transform_batch_size=512,
      predict_batch_size=1024, **kwargs):
    n_transforms = self.transformer.n_transforms
    matrix_scores = np.zeros(
        (int(len(x_data[0]) / n_transforms), n_transforms, n_transforms))
    x_pred = self.predict(x_data[0], batch_size=predict_batch_size)
    # TODO: paralelice this
    for t_ind in range(n_transforms):
      ind_x_pred_queal_to_t_ind = np.where(x_data[1] == t_ind)[0]
      matrix_scores[:, :, t_ind] += x_pred[ind_x_pred_queal_to_t_ind]
    return matrix_scores

  # TODO: implement pre-transofrmed, in-situ-all, efficient transforming
  def predict_matrix_and_dirichlet_score(self, x_train, x_eval,
      transform_batch_size=512, predict_batch_size=1024,
      **kwargs):
    n_transforms = self.transformer.n_transforms
    diri_scores = np.zeros(int(len(x_eval[0]) / n_transforms))
    matrix_scores_train = self.predict_matrix_score(
        x_train, transform_batch_size, predict_batch_size, **kwargs)
    del x_train
    matrix_scores_eval = self.predict_matrix_score(
        x_eval, transform_batch_size, predict_batch_size, **kwargs)
    del x_eval
    for t_ind in tqdm(range(n_transforms)):
      observed_dirichlet = matrix_scores_train[:, :, t_ind]
      x_eval_p = matrix_scores_eval[:, :, t_ind]
      diri_scores += dirichlet_utils.dirichlet_score(
          observed_dirichlet, x_eval_p)
    diri_scores /= n_transforms
    return matrix_scores_eval, diri_scores

  def get_scores_dict(self, train_data, eval_data,
      transform_batch_size=512, predict_batch_size=1024, **kwargs):
    matrix_scores, diri_scores = self.predict_matrix_and_dirichlet_score(
        train_data, eval_data, transform_batch_size, predict_batch_size,
        **kwargs)
    matrix_scores = matrix_scores / self.transformer.n_transforms
    scores_dict = {
      general_keys.DIRICHLET: diri_scores,
      general_keys.MATRIX_TRACE: np.trace(matrix_scores, axis1=1, axis2=2),
      general_keys.ENTROPY: -1 * score_functions.get_entropy(matrix_scores),
      general_keys.CROSS_ENTROPY: -1 * score_functions.get_xH(self.transformer,
                                                              matrix_scores),
      general_keys.MUTUAL_INFORMATION: score_functions.get_xy_mutual_info(
          matrix_scores)
    }
    return scores_dict

  # TODO: refactor, too long
  def evaluate_od(self, train_data, eval_data, y_eval, dataset_name, class_name,
      validation_data=None, transform_batch_size=512, predict_batch_size=1024,
      additional_score_save_path_list=None, save_hist_folder_path=None,
      **kwargs):
    print('evaluating')
    if validation_data is None:
      validation_data = eval_data
    # print('start eval')
    eval_scores_dict = self.get_scores_dict(
        train_data, eval_data, transform_batch_size, predict_batch_size,
        **kwargs)
    del eval_data
    # print('start val')
    validation_scores_dict = self.get_scores_dict(
        train_data, validation_data, transform_batch_size, predict_batch_size,
        **kwargs)
    del train_data, validation_data
    # print('start metric')
    metrics_of_each_score = {}
    for score_name, scores_value in eval_scores_dict.items():
      metrics_save_path = self.get_metrics_save_path(score_name, dataset_name,
                                                     class_name)
      metrics_of_each_score[score_name] = self.get_metrics_dict(
          scores_value, validation_scores_dict[score_name], y_eval,
          metrics_save_path)
      self._save_on_additional_paths(
          additional_score_save_path_list, metrics_save_path,
          metrics_of_each_score[score_name])
      self._save_histogram(metrics_of_each_score[score_name], score_name,
                           dataset_name, class_name, save_hist_folder_path)
    return metrics_of_each_score


if __name__ == '__main__':
  from parameters import loader_keys
  from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
  from modules.geometric_transform.transformations_tf import TransTransformer
  import time

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
  hits_outlier_dataset = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = hits_outlier_dataset.get_outlier_detection_datasets()

  transformer = TransTransformer()
  x_train_transform, y_train_transform = transformer.apply_all_transforms(
      x=x_train)
  x_val_transform, y_val_transform = transformer.apply_all_transforms(
      x=x_val)
  x_test_transform, y_test_transform = transformer.apply_all_transforms(
      x=x_test)
  time.sleep(10)
  mdl = AlreadyTransformODModel(transformer=transformer,
      input_shape=x_train.shape[1:])

  batch_size = 128
  mdl.fit(
      (x_train_transform, y_train_transform),
      (x_val_transform, y_val_transform), train_batch_size=batch_size,
      epochs=2  # int(np.ceil(200 / transformer.n_transforms))
  )
  met_dict = mdl.evaluate_od(
      (x_train_transform, y_train_transform),
      (x_test_transform, y_test_transform), y_test, 'hits-4-c', 'real',
      (x_val_transform, y_val_transform))

  print('\nroc_auc')
  for key in met_dict.keys():
    print(key, met_dict[key]['roc_auc'])

  """
  roc_auc
  dirichlet 0.9896582500000001
  matrix_trace 0.9541035
  entropy 0.9820515000000001
  cross_entropy 0.9614397499999999
  mutual_information 0.9889197499999999
  """
