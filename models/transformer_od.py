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
import numpy as np
from modules import dirichlet_utils, utils
from modules import score_functions
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from modules.metrics import accuracies_by_threshold, accuracy_at_thr
import pprint
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

"""In situ transformation perform"""


# TODO: think if its better to create a trainer instead of an encapsulated model
class TransformODModel(tf.keras.Model):
  def __init__(self, data_loader: ZTFOutlierLoader,
      transformer: AbstractTransformer, input_shape, depth=10,
      widen_factor=4, results_folder_name='', name='Transformer_OD_Model',
      **kwargs):
    super().__init__(name=name)
    self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    self.main_model_path = self.create_main_model_paths(results_folder_name,
                                                        self.name)
    utils.check_paths(self.main_model_path)
    self.data_loader = data_loader
    self.transformer = transformer
    self.network = self.get_network(
        input_shape=input_shape, n_classes=self.transformer.n_transforms,
        depth=depth, widen_factor=widen_factor, **kwargs)

  # TODO: do a param dict
  def get_network(self, input_shape, n_classes,
      depth, widen_factor, **kwargs):
    return WideResidualNetwork(
        input_shape=input_shape, n_classes=n_classes, depth=depth,
        widen_factor=widen_factor, **kwargs)

  def call(self, input_tensor, training=False):
    return self.network(input_tensor, training)

  def create_main_model_paths(self, results_folder_name, model_name):
    results_folder_path = self._results_folder_name_to_path(
        results_folder_name)
    main_model_path = os.path.join(results_folder_path, model_name)
    return main_model_path

  def _results_folder_name_to_path(self, result_folder_name):
    if 'results' in result_folder_name:
      return result_folder_name
    else:
      return os.path.join(PROJECT_PATH, 'results', result_folder_name)

  def create_specific_model_paths(self):
    self.specific_model_folder = os.path.join(self.main_model_path,
                                              self.transformer.name,
                                              '%s_%s' % (self.name, self.date))
    self.checkpoint_folder = os.path.join(self.specific_model_folder,
                                          'checkpoints')
    # self.tb_path = os.path.join(self.model_path, 'tb_summaries')
    utils.check_paths(
        [self.specific_model_folder, self.checkpoint_folder])

  def fit(self, x_train, x_val, transform_batch_size=512, train_batch_size=128,
      epochs=2, **kwargs):
    self.create_specific_model_paths()
    # ToDo: must be network? or just self.compile???
    self.network.compile(
        general_keys.ADAM, general_keys.CATEGORICAL_CROSSENTROPY,
        [general_keys.ACC])
    x_train_transform, y_train_transform = \
      self.transformer.apply_all_transforms(
          x=x_train, batch_size=transform_batch_size)
    x_val_transform, y_val_transform = \
      self.transformer.apply_all_transforms(
          x=x_val, batch_size=transform_batch_size)
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=0,
        restore_best_weights=True)
    self.network.fit(
        x=x_train_transform, y=tf.keras.utils.to_categorical(y_train_transform),
        validation_data=(
          x_val_transform, tf.keras.utils.to_categorical(y_val_transform)),
        batch_size=train_batch_size,
        epochs=epochs, callbacks=[es], **kwargs)
    weight_path = os.path.join(self.checkpoint_folder,
                               'final_weights.h5')
    del x_train, x_val, x_train_transform, x_val_transform, y_train_transform, y_val_transform
    self.save_weights(weight_path)

  def predict_dirichlet_score(self, x_train, x_eval,
      transform_batch_size=512, predict_batch_size=1024,
      **kwargs):
    _, diri_scores = self.predict_matrix_and_dirichlet_score(
        x_train, x_eval, transform_batch_size, predict_batch_size, **kwargs)
    return diri_scores

  # TODO: implement pre-transofrmed, in-situ-all, efficient transforming
  def predict_matrix_score(self, x, transform_batch_size=512,
      predict_batch_size=1024, **kwargs):
    n_transforms = self.transformer.n_transforms
    x_transformed, y_transformed = self.transformer.apply_all_transforms(
        x, transform_batch_size)
    x_pred = self.network.predict(x_transformed, batch_size=predict_batch_size)
    matrix_scores = np.zeros((len(x), n_transforms, n_transforms))
    # TODO: paralelice this
    for t_ind in range(n_transforms):
      ind_x_pred_queal_to_t_ind = np.where(y_transformed == t_ind)[0]
      matrix_scores[:, :, t_ind] += x_pred[ind_x_pred_queal_to_t_ind]
    return matrix_scores

  # TODO: implement pre-transofrmed, in-situ-all, efficient transforming
  def predict_matrix_and_dirichlet_score(self, x_train, x_eval,
      transform_batch_size=512, predict_batch_size=1024,
      **kwargs):
    n_transforms = self.transformer.n_transforms
    diri_scores = np.zeros(len(x_eval))
    matrix_scores = np.zeros((len(x_eval), n_transforms, n_transforms))
    for t_ind in tqdm(range(n_transforms)):
      x_train_transformed, _ = self.transformer.apply_transforms(
          x_train, [t_ind], transform_batch_size)
      observed_dirichlet = self.predict(
          x_train_transformed, batch_size=predict_batch_size, **kwargs)
      x_eval_transformed, _ = self.transformer.apply_transforms(
          x_eval, [t_ind], transform_batch_size)
      x_eval_p = self.predict(
          x_eval_transformed, batch_size=predict_batch_size, **kwargs)
      diri_scores += dirichlet_utils.dirichlet_score(
          observed_dirichlet, x_eval_p)
      matrix_scores[:, :, t_ind] += x_eval_p
      del x_train_transformed, x_eval_transformed
    diri_scores /= n_transforms
    return matrix_scores, diri_scores

  def get_scores_dict(self, x_train, x_eval,
      transform_batch_size=512, predict_batch_size=1024, **kwargs):
    matrix_scores, diri_scores = self.predict_matrix_and_dirichlet_score(
        x_train, x_eval, transform_batch_size, predict_batch_size, **kwargs)
    matrix_scores = matrix_scores / self.transformer.n_transforms
    scores_dict = {
      general_keys.DIRICHLET: diri_scores,
      general_keys.MATRIX_TRACE: np.trace(matrix_scores, axis1=1, axis2=2),
      general_keys.ENTROPY: -1 * score_functions.get_entropy(matrix_scores),
      general_keys.CROSS_ENTROPY: -1 * score_functions.get_xH(self.transformer,
                                                              matrix_scores)
    }
    return scores_dict

  # TODO: refactor, too long and include keys
  def get_metrics_save_path(self, score_name, dataset_name, class_name):
    results_name = self._get_score_result_name(score_name, dataset_name,
                                               class_name)
    results_file_name = '{}.npz'.format(results_name)
    all_results_folder = os.path.join(self.main_model_path, 'all_metric_files',
                                      dataset_name)
    utils.check_paths(all_results_folder)
    results_file_path = os.path.join(all_results_folder, results_file_name)
    return results_file_path

  # TODO: refactor, too long
  def evaluate_od(self, x_train, x_eval, y_eval, dataset_name, class_name,
      x_validation=None, transform_batch_size=512, predict_batch_size=1024,
      additional_score_save_path_list=None, save_hist_folder_path=None,
      **kwargs):
    print('evaluating')
    if x_validation is None:
      x_validation = x_eval
    # print('start eval')
    eval_scores_dict = self.get_scores_dict(
        x_train, x_eval, transform_batch_size, predict_batch_size, **kwargs)
    # print('start val')
    validation_scores_dict = self.get_scores_dict(
        x_train, x_validation, transform_batch_size, predict_batch_size,
        **kwargs)
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

  def _get_score_result_name(self, score_name, dataset_name,
      class_name):
    model_score_name = ('%s_%s' % (self.name, score_name)).replace("_", "-")
    dataset_plus_transformer_name = (
        '%s_%s' % (dataset_name, self.transformer.name)).replace("_", "-")
    results_name = '{}_{}_{}_{}'.format(
        dataset_plus_transformer_name, model_score_name, class_name, self.date)
    return results_name

  def _save_histogram(self, score_metric_dict, score_name, dataset_name,
      class_name, save_folder_path=None):
    if save_folder_path is None:
      return
    # TODO: refactor usage of percentile, include it in metrics and
    #  get it from key
    percentile = 95.46
    scores_val = score_metric_dict['scores_val']
    auc_roc = score_metric_dict['roc_auc']
    accuracies = score_metric_dict['accuracies']
    scores = score_metric_dict['scores']
    labels = score_metric_dict['labels']
    thresholds = score_metric_dict['roc_thresholds']
    accuracy_at_percentile = score_metric_dict['acc_at_percentil']
    inliers_scores = scores[labels == 1]
    outliers_scores = scores[labels != 1]
    min_score = np.min(scores)
    max_score = np.max(scores)
    thr_percentile = np.percentile(scores_val, 100 - percentile)
    fig = plt.figure(figsize=(8, 6))
    ax_hist = fig.add_subplot(111)
    ax_hist.set_title(
        'AUC_ROC: %.2f%%, BEST ACC: %.2f%%' % (
          auc_roc * 100, np.max(accuracies) * 100))
    ax_acc = ax_hist.twinx()
    hist1 = ax_hist.hist(inliers_scores, 100, alpha=0.5,
                         label='inlier', range=[min_score, max_score])
    hist2 = ax_hist.hist(outliers_scores, 100, alpha=0.5,
                         label='outlier', range=[min_score, max_score])
    _, max_ = ax_hist.set_ylim()
    ax_hist.set_ylabel('Counts', fontsize=12)
    ax_hist.set_xlabel(score_name, fontsize=12)
    # acc plot
    ax_acc.set_ylim([0.5, 1.0])
    ax_acc.yaxis.set_ticks(np.arange(0.5, 1.05, 0.05))
    ax_acc.set_ylabel('Accuracy', fontsize=12)
    acc_plot = ax_acc.plot(thresholds, accuracies, lw=2,
                           label='Accuracy by\nthresholds',
                           color='black')
    percentil_plot = ax_hist.plot([thr_percentile, thr_percentile], [0, max_],
                                  'k--',
                                  label='thr percentil %i on %s' % (
                                    percentile, dataset_name))
    ax_hist.text(thr_percentile,
                 max_ * 0.6,
                 'Acc: {:.2f}%'.format(accuracy_at_percentile * 100))
    ax_acc.grid(ls='--')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1),
               bbox_transform=ax_hist.transAxes)
    results_name = self._get_score_result_name(score_name, dataset_name,
                                               class_name)
    fig.savefig(
        os.path.join(save_folder_path, '%s_hist_thr_acc.png' % results_name),
        bbox_inches='tight')
    plt.close()

  # TODO: refactor to avoid usage of metrics_save_path,
  # this should be additional_paths_list and data
  def _save_on_additional_paths(self, additional_paths_list: list,
      metrics_save_path: str, metrics_dict: dict):
    if additional_paths_list is None:
      return
    if not isinstance(additional_paths_list, list):
      additional_paths_list = [additional_paths_list]
    for path in additional_paths_list:
      metric_file_name = os.path.basename(metrics_save_path)
      additional_save_path = os.path.join(path, metric_file_name)
      np.savez_compressed(additional_save_path, **metrics_dict)

  def get_metrics_dict(self, scores, scores_val, labels, save_file_path=None,
      percentile=95.46):
    scores = scores.flatten()
    labels = labels.flatten()
    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]
    truth = np.concatenate(
        (np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)
    accuracies = accuracies_by_threshold(labels, scores, roc_thresholds)
    # 100-percentile is necesary because normal data is at the right of anormal
    acc_at_percentil = accuracy_at_thr(
        labels, scores, np.percentile(scores_val, 100 - percentile))
    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(
        truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)
    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(
        truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)
    metrics_dict = {'scores': scores, 'labels': labels,
                    'scores_val': scores_val, 'fpr': fpr,
                    'tpr': tpr, 'roc_thresholds': roc_thresholds,
                    'roc_auc': roc_auc,
                    'precision_norm': precision_norm,
                    'recall_norm': recall_norm,
                    'pr_thresholds_norm': pr_thresholds_norm,
                    'pr_auc_norm': pr_auc_norm,
                    'precision_anom': precision_anom,
                    'recall_anom': recall_anom,
                    'pr_thresholds_anom': pr_thresholds_anom,
                    'pr_auc_anom': pr_auc_anom,
                    'accuracies': accuracies,
                    'max_accuracy': np.max(accuracies),
                    'acc_at_percentil': acc_at_percentil}
    if save_file_path is not None:
      # TODO: check if **metrics_dict works
      np.savez_compressed(save_file_path, **metrics_dict)
    else:
      pprint.pprint((metrics_dict))
    return metrics_dict


if __name__ == '__main__':
  from parameters import loader_keys
  from modules.geometric_transform.transformations_tf import Transformer

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
  model = TransformODModel(
      data_loader=ztf_od_loader, transformer=transformer,
      input_shape=x_train.shape[1:])
  model.build(tuple([None] + list(x_train.shape[1:])))
  weight_path = os.path.join(PROJECT_PATH, 'results', model.name,
                             'my_checkpoint.h5')
  model.load_weights(weight_path)
  # model.fit(x_train, x_val)

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
  # start_time = time.time()
  # met_dict = model.evaluate_od(
  #     x_train, x_test, y_test, 'ztf-real-bog-v1', 'real', x_val)
  # print(
  #     "Time model.evaluate_od %s" % utils.timer(
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
