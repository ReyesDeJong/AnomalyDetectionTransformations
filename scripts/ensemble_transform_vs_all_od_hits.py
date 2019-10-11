import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from keras.utils import to_categorical
from modules.data_loaders.base_line_loaders import load_hits

from transformations import TransTransformer
from models.simple_network import create_simple_network
import time
import datetime
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from tqdm import tqdm
from scripts.detached_transformer_od_hits import \
  plot_histogram_disc_loss_acc_thr
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras.layers import *
from keras.models import Model
import torch
import torch.nn as nn


def replicate_to_size(data_array, size):
  if len(data_array) < size:
    return replicate_to_size(np.concatenate([data_array, data_array]), size)
  else:
    size_left = size - len(data_array)
    return np.concatenate([data_array, data_array[:size_left]])


def plot_matrix_score(imgs, matrix_scores, labels=None, plot_inliers=True,
    n_to_plot=1):
  if isinstance(n_to_plot, int):
    if plot_inliers:
      inlier_idxs = np.where(labels == True)[0]
      plot_indxs = np.random.choice(inlier_idxs, n_to_plot, replace=False)
    else:
      outlier_idxs = np.where(labels == False)[0]
      plot_indxs = np.random.choice(outlier_idxs, n_to_plot, replace=False)
  else:
    plot_indxs = n_to_plot
  for indx in plot_indxs:
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(imgs[indx, ..., 0])
    ax[1].imshow(matrix_scores[indx])
    ax[1].set_title('Inlier: %s index: %i trace: %.4f' % (
      str(plot_inliers), indx, np.trace(matrix_scores[indx])))
    plt.show()


def get_entropy(matrix_scores, epsilon=1e-10):
  norm_scores = matrix_scores / np.sum(matrix_scores, axis=(1, 2))[
    ..., np.newaxis, np.newaxis]
  log_scores = np.log(norm_scores + epsilon)
  product = norm_scores * log_scores
  entropy = -np.sum(product, axis=(1, 2))
  return entropy


def get_list_of_models_without_softmax(model_list):
  short_model_list = []
  for model_i in model_list:
    logits = model_i.layers[-2].output
    short_model = Model(inputs=model_i.inputs, outputs=logits)
    short_model_list.append(short_model)
  return short_model_list


if __name__ == "__main__":
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
  sess = tf.Session(config=config)
  set_session(sess)

  single_class_ind = 1

  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_hits(
      n_samples_by_class=16000,
      test_size=0.25,
      val_size=0.125, return_val=True)
  print(x_train.shape)
  print(x_val.shape)
  print(x_test.shape)

  transformer = TransTransformer(8, 8)

  # get inliers of specific class
  x_train_task = x_train[y_train.flatten() == single_class_ind]
  print(x_train_task.shape)

  x_val_task = x_val[y_val.flatten() == single_class_ind]
  print(x_val_task.shape)
  # [0_i, ..., (N_transforms-1)_i, ..., ..., 0_N_samples, ...,
  # (N_transforms-1)_N_samples] shape: (N_transforms*N_samples,)
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

  models_list = []
  for transform_idx in range(transformer.n_transforms):
    print("Model %i" % transform_idx)
    mdl = create_simple_network(input_shape=x_train.shape[1:],
                                num_classes=2, dropout_rate=0.5)
    mdl.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['acc'])

    # separate inliers as an specific transform an the rest as outlier for an specific classifier, balance by replication
    selected_transformation_to_train = transform_idx
    # while transformation_to_train!=0:
    #   transformation_to_train = np.random.choice(transformations_inds_train, 1)

    selected_transform_indxs_train = \
      np.where(transformations_inds_train == selected_transformation_to_train)[
        0]
    non_transform_indxs_train = \
      np.where(transformations_inds_train != selected_transformation_to_train)[
        0]
    selected_transform_indxs_val = \
      np.where(transformations_inds_val == selected_transformation_to_train)[0]
    non_transform_indxs_val = \
      np.where(transformations_inds_val != selected_transformation_to_train)[0]

    oversampled_selected_trans_idxs_train = replicate_to_size(
        selected_transform_indxs_train, len(non_transform_indxs_train))
    # oversampled_selected_trans_idxs_val = replicate_to_size(
    #   selected_transform_indxs_val, len(non_transform_indxs_val))
    # val_x_binary = np.concatenate([x_val_task_transformed[oversampled_selected_trans_idxs_val],
    #                                  x_val_task_transformed[
    #                                    non_transform_indxs_val]])
    # val_y_binary = np.concatenate([np.ones_like(oversampled_selected_trans_idxs_val), np.zeros_like(non_transform_indxs_val)])

    subsamples_val_idxs = np.random.choice(non_transform_indxs_val,
                                           len(selected_transform_indxs_val),
                                           replace=False)

    train_x_binary = np.concatenate(
        [x_train_task_transformed[oversampled_selected_trans_idxs_train],
         x_train_task_transformed[non_transform_indxs_train]])
    train_y_binary = np.concatenate(
        [np.ones_like(oversampled_selected_trans_idxs_train),
         np.zeros_like(non_transform_indxs_train)])
    val_x_binary = np.concatenate(
        [x_val_task_transformed[selected_transform_indxs_val],
         x_val_task_transformed[
           subsamples_val_idxs]])
    val_y_binary = np.concatenate([np.ones_like(selected_transform_indxs_val),
                                   np.zeros_like(subsamples_val_idxs)])

    print('Train_size: ', np.unique(train_y_binary, return_counts=True))
    print('Val_size: ', np.unique(val_y_binary, return_counts=True))

    # plt.imshow(train_x_binary[0, ..., 0])
    # plt.show()
    # plt.imshow(train_x_binary[len(oversampled_selected_trans_idxs_train), ..., 0])
    # plt.show()

    start_time = time.time()
    mdl.fit(x=train_x_binary, y=to_categorical(train_y_binary),
            validation_data=(val_x_binary, to_categorical(val_y_binary)),
            batch_size=batch_size,
            epochs=1,  # int(np.ceil(200 / transformer.n_transforms))
            )
    time_usage = str(datetime.timedelta(
        seconds=int(round(time.time() - start_time))))
    print("Time to train model: " + time_usage)
    models_list.append(mdl)

  print('N models: ', transform_idx + 1)
  # testing model on other data
  # models_list[0].evaluate(x=val_x_binary, y=to_categorical(val_y_binary))

  print('test_size: ', np.unique(y_test, return_counts=True))
  test_out_idxs = np.where(y_test == 0)[0]
  test_out_x = x_test[test_out_idxs]
  plt.imshow(test_out_x[np.random.randint(len(test_out_x)), ..., 0])
  plt.show()
  test_in_idxs = np.where(y_test == 1)[0]
  test_in_x = x_test[test_in_idxs]
  plt.imshow(test_in_x[np.random.randint(len(test_in_x)), ..., 0])
  plt.show()

  bins = 100
  plt.hist(models_list[0].predict(test_in_x)[:, 1], bins=bins)
  plt.title('inliers')
  plt.show()
  plt.hist(models_list[0].predict(test_out_x)[:, 1], bins=bins)
  plt.title('outliers')
  plt.show()

  # Get scores
  plain_scores = np.zeros((len(x_test),))
  for t_ind in tqdm(range(transformer.n_transforms)):
    # predictions for a single transformation
    x_test_p = models_list[t_ind].predict(
        transformer.transform_batch(x_test, [t_ind] * len(x_test)),
        batch_size=1024)
    plain_scores += x_test_p[:, 1]

  plain_scores /= transformer.n_transforms
  labels = y_test.flatten() == single_class_ind

  bins = 100
  plt.hist(plain_scores[test_in_idxs], bins=bins)
  plt.title('scores_inliers')
  plt.show()
  plt.hist(plain_scores[test_out_idxs], bins=bins)
  plt.title('scores_outliers')
  plt.show()

  scores_pos = plain_scores[labels == 1]
  scores_neg = plain_scores[labels != 1]
  bins = 100
  plt.hist(scores_pos, bins=bins)
  plt.title('scores_inliers')
  plt.show()
  plt.hist(scores_neg, bins=bins)
  plt.title('scores_outliers')
  plt.show()

  truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
  preds = np.concatenate((scores_neg, scores_pos))
  fpr, tpr, roc_thresholds = roc_curve(truth, preds)
  roc_auc = auc(fpr, tpr)
  print(roc_auc)

  plot_histogram_disc_loss_acc_thr(plain_scores[labels], plain_scores[~labels],
                                   x_label_name='EnsembleTransformations_scores_hits',
                                   path='../results')

  # Scores arcsinh
  plain_neg_scores = 1 - plain_scores
  plain_norm_scores = plain_neg_scores - np.min(plain_neg_scores)
  plain_norm_scores = plain_norm_scores / plain_norm_scores.max()
  plain_arcsinh_scores = np.arcsinh(plain_norm_scores * 1000000)

  plot_histogram_disc_loss_acc_thr(plain_arcsinh_scores[labels],
                                   plain_arcsinh_scores[~labels],
                                   path='../results',
                                   x_label_name='EnsembleTransformations_arcsinh*10000_scores_hits')

  ## matrices
  # transform test
  transformations_inds_test = np.tile(np.arange(transformer.n_transforms),
                                      len(x_test))
  start_time = time.time()
  x_test_transformed = transformer.transform_batch(
      np.repeat(x_test, transformer.n_transforms, axis=0),
      transformations_inds_test)
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time to perform transforms: " + time_usage)

  # Get matrix scores
  matrix_scores = np.zeros(
      (len(x_test), transformer.n_transforms, transformer.n_transforms))
  for model_t_ind in tqdm(range(transformer.n_transforms)):
    for t_ind in range(transformer.n_transforms):
      test_specific_transform_indxs = np.where(
          transformations_inds_test == t_ind)
      x_test_specific_transform = x_test_transformed[
        test_specific_transform_indxs]
      # predictions for a single transformation
      x_test_p = models_list[model_t_ind].predict(x_test_specific_transform,
                                                  batch_size=64)
      matrix_scores[:, model_t_ind, t_ind] += x_test_p[:, 1]

  matrix_scores /= transformer.n_transforms
  labels = y_test.flatten() == single_class_ind

  # plot_matrix_score(x_test, matrix_scores, labels, plot_inliers=True,
  #                   n_to_plot=15)
  # plot_matrix_score(x_test, matrix_scores, labels, plot_inliers=False,
  #                   n_to_plot=15)

  entropy_scores = get_entropy(matrix_scores)
  plot_histogram_disc_loss_acc_thr(entropy_scores[labels],
                                   entropy_scores[~labels],
                                   path='../results',
                                   x_label_name='EnsembleTransformations_entropy_scores_hits')

  # Get scores for xentropy
  matrix_scores_2class = np.zeros(
      (len(x_test), transformer.n_transforms, transformer.n_transforms, 2))
  for model_t_ind in tqdm(range(transformer.n_transforms)):
    for t_ind in range(transformer.n_transforms):
      test_specific_transform_indxs = np.where(
          transformations_inds_test == t_ind)
      x_test_specific_transform = x_test_transformed[
        test_specific_transform_indxs]
      # predictions for a single transformation
      x_test_p = models_list[model_t_ind].predict(x_test_specific_transform,
                                                  batch_size=64)
      matrix_scores_2class[:, model_t_ind, t_ind] += x_test_p

  matrix_scores_2class /= transformer.n_transforms
  labels = y_test.flatten() == single_class_ind

  # plot_matrix_score(x_test, matrix_scores_2class[...,1], labels, plot_inliers=True,
  #                   n_to_plot=5)
  # plot_matrix_score(x_test, matrix_scores_2class[...,1], labels, plot_inliers=False,
  #                   n_to_plot=5)
  # plot_matrix_score(x_test, matrix_scores_2class[...,0], labels, plot_inliers=True,
  #                   n_to_plot=5)
  # plot_matrix_score(x_test, matrix_scores_2class[...,0], labels, plot_inliers=False,
  #                   n_to_plot=5)

  # logits = models_list[0].layers[-2].output
  # short_model = Model(inputs=models_list[0].inputs, outputs=logits)
  short_models_list = get_list_of_models_without_softmax(models_list)
  # Get logits for xentropy
  matrix_logits = np.zeros(
      (len(x_test), transformer.n_transforms, transformer.n_transforms, 2))
  for model_t_ind in tqdm(range(transformer.n_transforms)):
    for t_ind in range(transformer.n_transforms):
      test_specific_transform_indxs = np.where(
          transformations_inds_test == t_ind)
      x_test_specific_transform = x_test_transformed[
        test_specific_transform_indxs]
      # predictions for a single transformation
      x_test_p = short_models_list[model_t_ind].predict(
        x_test_specific_transform,
        batch_size=64)
      matrix_logits[:, model_t_ind, t_ind] += x_test_p

  labels = y_test.flatten() == single_class_ind
  # plot_matrix_score(x_test, matrix_logits[..., 1], labels, plot_inliers=True,
  #                   n_to_plot=5)

  # logits_matrix_ph = tf.placeholder(
  #     dtype=tf.float32,
  #     shape=(None, transformer.n_transforms, transformer.n_transforms, 2))
  # eye_tf = tf.constant(np.eye(transformer.n_transforms))
  # gt_matrix = tf.stack([eye_tf] * len(matrix_logits))
  # # cross_entropy

  xH = nn.CrossEntropyLoss(reduction='none')
  gt_matrix = np.stack([np.eye(transformer.n_transforms)] * len(matrix_logits))
  gt_torch = torch.LongTensor(gt_matrix)


  matrix_logits_torch = torch.FloatTensor(np.swapaxes(np.swapaxes(matrix_logits,1,-1),-1,-2))
  loss_xH = xH(matrix_logits_torch, gt_torch)
  batch_xH = np.mean(loss_xH.numpy(), axis=(-1,-2))

  plot_histogram_disc_loss_acc_thr(batch_xH[labels],
                                   batch_xH[~labels],
                                   path='../results',
                                   x_label_name='EnsembleTransformations_xH_scores_hits')


  # get worst n traces for inliers and best n traces for outliers

  in_matrix_score = matrix_scores[labels]
  out_matrix_score = matrix_scores[~labels]
  indx_in = np.argsort(np.trace(in_matrix_score, axis1=1, axis2=2))
  indx_out = np.argsort(np.trace(out_matrix_score, axis1=1, axis2=2))
  x_test_in = x_test[labels]
  x_test_out = x_test[~labels]

  #worst in
  plot_matrix_score(x_test_in, in_matrix_score, n_to_plot=indx_in[:5], plot_inliers=True)
  # worst outliers out (high trace)
  plot_matrix_score(x_test_out, out_matrix_score, n_to_plot=indx_out[-5:], plot_inliers=False)
  # best outliers out (low trace)
  plot_matrix_score(x_test_out, out_matrix_score, n_to_plot=indx_out[:5], plot_inliers=False)
