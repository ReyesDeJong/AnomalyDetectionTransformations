import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import psi, polygamma
from keras.utils import to_categorical
from modules.data_loaders.base_line_loaders import load_hits

from transformations import Transformer
from models.wide_residual_network import create_wide_residual_network
import time
import datetime
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import auc


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

  single_class_ind = 1

  (x_train, y_train), (x_test, y_test) = load_hits(n_samples_by_class=16000,
                                                   test_size=0.25,
                                                   val_size=0.125)
  print(x_train.shape)
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
  # [0_i, ..., (N_transforms-1)_i, ..., ..., 0_N_samples, ...,
  # (N_transforms-1)_N_samples] shape: (N_transforms*N_samples,)
  transformations_inds = np.tile(np.arange(transformer.n_transforms),
                                 len(x_train_task))
  print(len(transformations_inds))
  #
  start_time = time.time()
  x_train_task_transformed = transformer.transform_batch(
      np.repeat(x_train_task, transformer.n_transforms, axis=0),
      transformations_inds)
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time to perform transforms: " + time_usage)
  print(x_train_task_transformed.shape)
  batch_size = 128

  start_time = time.time()
  mdl.fit(x=x_train_task_transformed, y=to_categorical(transformations_inds),
          batch_size=batch_size,
          epochs=int(np.ceil(200 / transformer.n_transforms))
          )
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time to train model: " + time_usage)

  scores = np.zeros((len(x_test),))
  observed_data = x_train_task

  # # testing inside for
  # t_ind = np.random.randint(transformer.n_transforms)
  # observed_dirichlet = mdl.predict(
  #     transformer.transform_batch(observed_data, [t_ind] * len(observed_data)),
  #     batch_size=1024)
  # predicted_labels = np.argmax(observed_dirichlet, axis=-1)
  # print('index to predict: ', t_ind, '\nPredicted counts: ',
  #       np.unique(predicted_labels, return_counts=True))
  # log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)
  # print('log_p_hat_train.shape: ', log_p_hat_train.shape)
  # alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
  # print('alpha_sum_approx.shape: ', alpha_sum_approx.shape)
  # alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx
  # print('alpha_0.shape: ', alpha_0.shape)
  # mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
  # print('mle_alpha_t.shape: ', mle_alpha_t.shape)
  # x_test_p = mdl.predict(
  #     transformer.transform_batch(x_test, [t_ind] * len(x_test)),
  #     batch_size=1024)
  # predicted_test_labels = np.argmax(x_test_p, axis=-1)
  # print('index to predict: ', t_ind, '\nPredicted test counts: ',
  #       np.unique(predicted_test_labels, return_counts=True))
  #
  # score_for_specific_transform = dirichlet_normality_score(mle_alpha_t,
  #                                                          x_test_p)
  # print('score_for_specific_transform.shape: ',
  #       score_for_specific_transform.shape)

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
    scores += dirichlet_normality_score(mle_alpha_t, x_test_p)

  scores /= transformer.n_transforms
  labels = y_test.flatten() == single_class_ind

  plot_histogram_disc_loss_acc_thr(scores[labels], scores[~labels],
                                   path='../results',
                                   x_label_name='Transformations_Dscores_hits')

  # Dirichlet transforms with arcsin
  neg_scores = -scores
  norm_scores = neg_scores - np.min(neg_scores)
  norm_scores = norm_scores / np.max(norm_scores)
  arcsinh_scores = np.arcsinh(norm_scores * 10000)
  inlier_arcsinh_score = arcsinh_scores[labels]
  outlier_arcsinh_score = arcsinh_scores[~labels]
  plot_histogram_disc_loss_acc_thr(inlier_arcsinh_score, outlier_arcsinh_score,
                                   '../results',
                                   'Transformations_arcsinh*10000_Dscores')

  # Transforms without dirichlet
  plain_scores = np.zeros((len(x_test),))
  for t_ind in tqdm(range(transformer.n_transforms)):
    # predictions for a single transformation
    x_test_p = mdl.predict(
        transformer.transform_batch(x_test, [t_ind] * len(x_test)),
        batch_size=1024)
    plain_scores += x_test_p[:, t_ind]

  plain_scores /= transformer.n_transforms
  labels = y_test.flatten() == single_class_ind

  plot_histogram_disc_loss_acc_thr(plain_scores[labels], plain_scores[~labels],
                                   path='../results',
                                   x_label_name='Transformations_scores_hits')

  # Transforms without dirichlet arcsinh
  plain_neg_scores = 1 - plain_scores
  plain_norm_scores = plain_neg_scores - np.min(plain_neg_scores)
  plain_norm_scores = plain_norm_scores / plain_norm_scores.max()
  plain_arcsinh_scores = np.arcsinh(plain_norm_scores * 10000)

  plot_histogram_disc_loss_acc_thr(plain_arcsinh_scores[labels],
                                   plain_arcsinh_scores[~labels],
                                   path='../results',
                                   x_label_name='Transformations_arcsinh*10000_scores_hits')
