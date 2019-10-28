#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:05:40 2018

Dataset Object

CHECK MAX DISBALANCE OPN REPLICATION FOR MULTICLASS

@author: ereyes
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Counter

#Todo: refactor metadata
class Dataset(object):
  """
  Constructor
  """

  def __init__(self, data_array, data_label, batch_size, meta_data=None):
    self.batch_counter = 0
    self.batch_counter_val = 0
    self.batch_size = batch_size
    self.data_array = np.array(data_array)
    self.data_label = np.array(self._check_data_labels(data_label))
    self.meta_data = self._check_meta_data(meta_data)

  def _check_meta_data(self, meta_data):
    if meta_data is None:
      return np.zeros(self.data_array.shape[0])
    return meta_data

  def _check_data_labels(self, data_labels):
    if data_labels is None:
      return np.zeros(self.data_array.shape[0])
    return data_labels

  def _merge_with_dataset(self, array, label_value):
    self.data_label = np.concatenate(
        (self.data_label, np.full(array.shape[0], label_value)))
    self.data_array = np.concatenate((self.data_array, array))

  def get_batch_images(self):
    batch, _ = self.get_batch()

    return batch

  def get_batch(self):
    if (self.batch_counter + self.batch_size < self.data_array.shape[0]):
      batch_image = self.data_array[
                    self.batch_counter:self.batch_counter + self.batch_size,
                    ...]
      batch_label = self.data_label[
                    self.batch_counter:self.batch_counter + self.batch_size,
                    ...]
      self.batch_counter += self.batch_size
      # print(get_batch.BATCH_COUNTER)
    else:
      self.batch_counter = 0
      self.shuffle_data()
      batch_image = self.data_array[
                    self.batch_counter:self.batch_counter + self.batch_size,
                    ...]
      batch_label = self.data_label[
                    self.batch_counter:self.batch_counter + self.batch_size,
                    ...]
      self.batch_counter += self.batch_size

    return batch_image, batch_label

  def get_batch_eval(self):
    if (self.batch_counter_val + self.batch_size < self.data_array.shape[0]):
      batch_image = self.data_array[
                    self.batch_counter_val:self.batch_counter_val + self.batch_size,
                    ...]
      batch_label = self.data_label[
                    self.batch_counter_val:self.batch_counter_val + self.batch_size,
                    ...]
      self.batch_counter_val += self.batch_size
      # print(get_batch.BATCH_COUNTER)
    else:
      left_samples = self.data_array.shape[0] - self.batch_counter_val
      batch_image = self.data_array[
                    self.batch_counter_val:self.batch_counter_val + left_samples,
                    ...]
      batch_label = self.data_label[
                    self.batch_counter_val:self.batch_counter_val + left_samples,
                    ...]
      self.batch_counter_val = 0

    return batch_image, batch_label

  def shuffle_data(self):
    idx = np.arange(self.data_array.shape[0])
    np.random.shuffle(idx)
    self.data_array = self.data_array[idx, ...]
    self.data_label = self.data_label[idx, ...]

  # TODO aboid data replication
  def balance_data_by_replication(self):
    # sort labels by quantity
    labels_count_sorted = np.argsort(list(Counter(self.data_label).values()))[
                          ::-1]
    label_values = np.array(list(Counter(self.data_label).keys())).astype(int)[
      labels_count_sorted]  # np.unique(self.data_label)
    for label_idx in range(label_values.shape[0] - 1):
      if label_idx == 0:
        first_labels_idx = np.where(self.data_label == label_values[0])[0]
        second_labels_idx = np.where(self.data_label == label_values[1])[0]
        data_set_train = Dataset(data_array=np.concatenate(
            [self.data_array[first_labels_idx],
             self.data_array[second_labels_idx]]),
                                 data_label=np.concatenate(
                                     [self.data_label[first_labels_idx],
                                      self.data_label[second_labels_idx]]),
                                 batch_size=8)
        data_set_train.balance_data_by_replication_2_classes()
        aux_balanced_data_array = data_set_train.data_array
        aux_balanced_data_label = data_set_train.data_label
      else:
        # label_idx+1 because it will be equal to one at first and we want to skip idx 1 considered above
        aux_labels_idx = \
        np.where(aux_balanced_data_label == label_values[label_idx])[0]
        remaining_labels_idx = \
        np.where(self.data_label == label_values[label_idx + 1])[0]

        data_set_train = Dataset(data_array=np.concatenate(
            [aux_balanced_data_array[aux_labels_idx],
             self.data_array[remaining_labels_idx]]),
                                 data_label=np.concatenate(
                                     [aux_balanced_data_label[aux_labels_idx],
                                      self.data_label[remaining_labels_idx]]),
                                 batch_size=8)
        data_set_train.balance_data_by_replication_2_classes()

        # get idx of remaing labels just replicated
        balanced_remaining_labels_idx = \
        np.where(data_set_train.data_label == label_values[label_idx + 1])[0]
        aux_balanced_data_array = np.concatenate([aux_balanced_data_array,
                                                  data_set_train.data_array[
                                                    balanced_remaining_labels_idx]])
        aux_balanced_data_label = np.concatenate([aux_balanced_data_label,
                                                  data_set_train.data_label[
                                                    balanced_remaining_labels_idx]])
    self.data_array = aux_balanced_data_array
    self.data_label = aux_balanced_data_label

  # TODO: change both values for uique functions (AVOID CODE REPLICATION)
  # TODO: recursively? replicate_data should be?
  # TODO: min_lbl_count changes on very iteration, it should stay the same or shuffle
  # of replicate_data cannot be
  def balance_data_by_replication_2_classes(self):
    max_disbalance = self.get_max_disbalance()
    max_lbl_count, min_lbl_count = self.get_max_min_label_count()
    max_lbl, min_lbl = self.get_max_min_label()

    if (max_disbalance == 0):
      return
    while (max_disbalance != 0):
      if (min_lbl_count > max_disbalance):
        self.replicate_data(min_lbl, max_disbalance)
        # max_disbalance = 0
      else:
        self.replicate_data(min_lbl, min_lbl_count)
        # max_disbalance -= min_lbl_count
      max_disbalance = self.get_max_disbalance()  #
    self.balance_data_by_replication_2_classes()
    return

  def get_max_disbalance(self):
    max_label_count, min_label_count = self.get_max_min_label_count()
    return max_label_count - min_label_count

  def get_max_min_label_count(self):
    max_label, min_label = self.get_max_min_label()

    max_label_count = np.where(self.data_label == max_label)[0].shape[0]
    min_label_count = np.where(self.data_label == min_label)[0].shape[0]

    return max_label_count, min_label_count

  def get_max_min_label(self):
    labels = np.unique(self.data_label)
    labels_count = []

    for j in range(labels.shape[0]):
      label_j_count = np.where(self.data_label == labels[j])[0].shape[0]
      labels_count.append(label_j_count)

    labels_count = np.array(labels_count)

    max_label = labels[np.where(labels_count == np.max(labels_count))[0][0]]
    min_label = labels[np.where(labels_count == np.min(labels_count))[0][0]]
    return max_label, min_label

  def replicate_data(self, label, samples_number):
    # print("%i samples replicated of class %i" %(samples_number,label))
    label_idx = np.where(self.data_label == label)[0]
    # np.random.shuffle(label_idx)
    label_idx = label_idx[0:samples_number]
    replicated_data_array = self.data_array[label_idx, ...]
    self._merge_with_dataset(replicated_data_array, label)

  def get_array_from_label(self, label):
    label_idx = np.where(self.data_label == label)[0]
    return self.data_array[label_idx]
