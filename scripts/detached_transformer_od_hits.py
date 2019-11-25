import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import psi, polygamma
from tensorflow.keras.utils import to_categorical
from modules.data_loaders.base_line_loaders import load_hits

from transformations import Transformer
from models.wide_residual_network import create_wide_residual_network
import time
import datetime
# from tensorflow.keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import auc
from modules.utils import check_paths

def calc_approx_alpha_sum(observations):
  N = len(observations)
  f = np.mean(observations, axis=0)

  return (N * (len(f) - 1) * (-psi(1))) / (
      N * np.sum(f * np.log(f)) - np.sum(
      f * np.sum(np.log(observations), axis=0)))


def inv_psi(y, iters=5):
  # initial estimate
  cond = y >= -2.22
  x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / (y - psi(1))

  for _ in range(iters):
    x = x - (psi(x) - y) / polygamma(1, x)
  return x


def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
  alpha_new = alpha_old = alpha_init
  for _ in range(max_iter):
    alpha_new = inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
    if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9:
      break
    alpha_old = alpha_new
  return alpha_new


def dirichlet_normality_score(alpha, p):
  return np.sum((alpha - 1) * np.log(p), axis=-1)


def plot_histogram_disc_loss_acc_thr(inliers_scores, outliers_scores,
    path=None, x_label_name='score', show=True, balance_classes=False,
    percentil=95.46, val_inliers_score=None):
  set_name = 'val'
  if val_inliers_score is None:
    val_inliers_score = inliers_scores
    set_name = 'test'
  if balance_classes:
    print('balancing outlier and inliers')
    if len(inliers_scores) != len(outliers_scores):
      print('There are %i inliers and %i outliers' % (
        len(inliers_scores), len(outliers_scores)))
      if len(inliers_scores) < len(outliers_scores):
        outliers_scores = np.random.choice(outliers_scores, len(inliers_scores),
                                           replace=False)
      else:
        inliers_scores = np.random.choice(inliers_scores, len(outliers_scores),
                                          replace=False)
    else:
      print('they are balanced')

  mean_inliers = np.mean(inliers_scores)
  mean_outliers = np.mean(outliers_scores)
  if mean_outliers < mean_inliers:
    print('scores *-1 to get in<out')
    inliers_scores = inliers_scores * -1
    outliers_scores = outliers_scores * -1
    val_inliers_score = val_inliers_score * -1

  thresholds = np.unique(np.concatenate([inliers_scores, outliers_scores]))

  accuracies = []
  tpr_list = []
  fpr_list = []
  for thr in thresholds[::-1]:
    FP = np.sum(outliers_scores < thr)
    TP = np.sum(inliers_scores < thr)
    TN = np.sum(outliers_scores >= thr)
    FN = np.sum(inliers_scores >= thr)

    accuracy = (TP + TN) / (FP + TP + TN + FN)
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    tpr_list.append(tpr)
    fpr_list.append(fpr)
    accuracies.append(accuracy)

  auc_roc = auc(fpr_list, tpr_list)

  min = np.min(np.concatenate([inliers_scores, outliers_scores]))
  max = np.max(np.concatenate([inliers_scores, outliers_scores]))

  # percentil 96 thr
  thr = np.percentile(val_inliers_score, percentil)
  scores_preds = (np.concatenate(
      [inliers_scores, outliers_scores]) < thr) * 1
  accuracy_at_percentil = np.mean(scores_preds == np.concatenate(
      [np.ones_like(inliers_scores), np.zeros_like(outliers_scores)]))

  fig = plt.figure(figsize=(8, 6))
  ax_hist = fig.add_subplot(111)
  ax_hist.set_title(
      'AUC_ROC: %.2f%%, BEST ACC: %.2f%%' % (
        auc_roc * 100, np.max(accuracies) * 100))
  ax_acc = ax_hist.twinx()
  hist1 = ax_hist.hist(inliers_scores, 100, alpha=0.5,
                       label='inlier', range=[min, max])
  hist2 = ax_hist.hist(outliers_scores, 100, alpha=0.5,
                       label='outlier', range=[min, max])
  _, max_ = ax_hist.set_ylim()
  ax_hist.set_ylabel('Counts', fontsize=12)
  ax_hist.set_xlabel(x_label_name, fontsize=12)

  ax_acc.set_ylim([0.5, 1.0])
  ax_acc.yaxis.set_ticks(np.arange(0.5, 1.05, 0.05))
  ax_acc.set_ylabel('Accuracy', fontsize=12)
  acc_plot = ax_acc.plot(thresholds[::-1], accuracies, lw=2,
                         label='Accuracy by\nthresholds',
                         color='black')
  percentil_plot = ax_hist.plot([thr, thr], [0, max_], 'k--',
                                label='thr percentil %i on %s' % (
                                percentil, set_name))
  ax_hist.text(thr,
               max_ * 0.6,
               'Acc: {:.2f}%'.format(accuracy_at_percentil * 100))

  ax_acc.grid(ls='--')
  fig.legend(loc="upper right", bbox_to_anchor=(1, 1),
             bbox_transform=ax_hist.transAxes)
  if path:
    fig.savefig(os.path.join(path, '%s_hist_thr_acc.png' % x_label_name),
                bbox_inches='tight')
  if show:
    plt.show()
  else:
    plt.close()


if __name__ == "__main__":
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
  sess = tf.Session(config=config)
  set_session(sess)
  save_path = '../results/Transforms_hits'
  check_paths(save_path)


  single_class_ind = 1

  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_hits(
      n_samples_by_class=10000,
      test_size=0.20,
      val_size=0.10, return_val=True, channels_to_get=[2])
  print(x_train.shape)
  print(x_val.shape)
  print(x_test.shape)

  transformer = Transformer(8, 8)
  n, k = (10, 4)

  mdl = create_wide_residual_network(input_shape=x_train.shape[1:],
                                     num_classes=transformer.n_transforms,
                                     depth=n, widen_factor=k)
  mdl.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])

  print(mdl.summary())

  # get inliers of specific class
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  print(x_train_task.shape)

  x_val_task = x_val[y_val.flatten() == single_class_ind]
  print(x_val_task.shape)

  transformations_inds_train = np.tile(np.arange(transformer.n_transforms),
                                       len(x_train_task))
  transformations_inds_val = np.tile(np.arange(transformer.n_transforms),
                                     len(x_val_task))
  print(len(transformations_inds_train))
  print(len(transformations_inds_val))

  # transform data
  start_time = time.time()
  x_train_task_transformed = transformer.transform_batch(
      np.repeat(x_train_task, transformer.n_transforms, axis=0),
      transformations_inds_train)
  x_val_task_transformed = transformer.transform_batch(
      np.repeat(x_val_task, transformer.n_transforms, axis=0),
      transformations_inds_val)
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time to perform transforms: " + time_usage)
  print(x_train_task_transformed.shape)
  print(x_val_task_transformed.shape)
  batch_size = 128

  start_time = time.time()
  mdl.fit(x=x_train_task_transformed, y=to_categorical(transformations_inds_train),
          batch_size=batch_size,
          epochs= int(np.ceil(200 / transformer.n_transforms))
          )
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time to train model: " + time_usage)

  test_scores = np.zeros((len(x_test),))
  val_scores_in = np.zeros((len(x_val_task),))
  observed_data = x_train_task

  # Dirichlet transforms
  for t_ind in tqdm(range(transformer.n_transforms)):
    # predictions for a single transformation
    observed_dirichlet = mdl.predict(
        transformer.transform_batch(observed_data,
                                    [t_ind] * len(observed_data)),
        batch_size=1024)
    log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

    alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
    alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

    mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)

    x_test_p = mdl.predict(
        transformer.transform_batch(x_test, [t_ind] * len(x_test)),
        batch_size=1024)
    test_scores += dirichlet_normality_score(mle_alpha_t, x_test_p)

  test_scores /= transformer.n_transforms

  # val
  # Dirichlet transforms
  for t_ind in tqdm(range(transformer.n_transforms)):
    # predictions for a single transformation
    observed_dirichlet = mdl.predict(
        transformer.transform_batch(observed_data,
                                    [t_ind] * len(observed_data)),
        batch_size=1024)
    log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

    alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
    alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

    mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)

    x_val_p = mdl.predict(
        transformer.transform_batch(x_val_task, [t_ind] * len(x_val_task)),
        batch_size=1024)
    val_scores_in += dirichlet_normality_score(mle_alpha_t, x_val_p)

  val_scores_in /= transformer.n_transforms

  labels = y_test.flatten() == single_class_ind

  plot_histogram_disc_loss_acc_thr(test_scores[labels], test_scores[~labels],
                                   path=save_path,
                                   x_label_name='Transformations_Dscores_hits', val_inliers_score=val_scores_in)


  # Transforms without dirichlet
  plain_scores_test = np.zeros((len(x_test),))
  for t_ind in tqdm(range(transformer.n_transforms)):
    # predictions for a single transformation
    x_test_p = mdl.predict(
        transformer.transform_batch(x_test, [t_ind] * len(x_test)),
        batch_size=1024)
    plain_scores_test += x_test_p[:, t_ind]

  plain_scores_test /= transformer.n_transforms
  #val
  plain_scores_val = np.zeros((len(x_val_task),))
  for t_ind in tqdm(range(transformer.n_transforms)):
    # predictions for a single transformation
    x_val_p = mdl.predict(
        transformer.transform_batch(x_val_task, [t_ind] * len(x_val_task)),
        batch_size=1024)
    plain_scores_val += x_val_p[:, t_ind]

  plain_scores_val /= transformer.n_transforms

  labels = y_test.flatten() == single_class_ind

  plot_histogram_disc_loss_acc_thr(plain_scores_test[labels], plain_scores_test[~labels],
                                   path=save_path,
                                   x_label_name='Transformations_scores_hits', val_inliers_score=plain_scores_val)

