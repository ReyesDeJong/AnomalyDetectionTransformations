import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from parameters import param_keys, general_keys
from modules.data_set_generic import Dataset
from modules.data_loaders.frame_to_input import FrameToInput
import pandas as pd
import numpy as np
from modules import utils


if __name__ == "__main__":
  random_seed = 42
  val_inlier_percentage = 0.1
  data_name = 'ztf_stamp_clf_for_geotrf_22-07-2020.pkl'
  data_folder = "/home/ereyes/Projects/Thesis/datasets"
  save_folder = '/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/'
  utils.check_path(save_folder)
  data_path = os.path.join(data_folder, data_name)

  ztf_dataset = pd.read_pickle(data_path)
  train_set = ztf_dataset['Train']
  validation_set = ztf_dataset['Validation']
  test_set = ztf_dataset['Test']
  # # dict_keys(['images', 'labels', 'features'])
  # print(train_set.keys())
  # print(train_set['features'])

  stamps = np.concatenate([train_set['images'], validation_set['images'],
                           test_set['images']])
  labels = np.concatenate(
      [train_set['labels'], validation_set['labels'], test_set['labels']])
  features = np.concatenate(
      [train_set['features'], validation_set['features'], test_set['features']])
  print('all data Set: ', np.unique(labels, return_counts=True))

  # Bogus dataset build
  bogus_stamps = stamps[labels == 4]
  inliers_stamps = stamps[labels != 4]


  inlier_indexes = np.arange(len(inliers_stamps))
  np.random.RandomState(seed=random_seed).shuffle(inlier_indexes)
  test_inliers = inliers_stamps[inlier_indexes[:len(bogus_stamps)]]
  remaining_inliers = inliers_stamps[inlier_indexes[len(bogus_stamps):]]
  remaining_inlier_indexes = np.arange(len(remaining_inliers))
  val_inliers = remaining_inliers[
                :int(len(remaining_inliers)*val_inlier_percentage)]
  train_inliers = remaining_inliers[
                  int(len(remaining_inliers)*val_inlier_percentage):]
  print('Inliers numbers Train %i Val %i Test %i' % (
      len(train_inliers), len(val_inliers), len(test_inliers)))

  x_test = np.concatenate([test_inliers, bogus_stamps])
  y_test = np.concatenate(
      [np.ones(len(test_inliers)), np.zeros(len(bogus_stamps))])
  print('\nTest set: ',
        np.unique(y_test, return_counts=True), '\n')

  x_val = val_inliers
  y_val = np.ones(len(val_inliers))
  x_train = train_inliers
  y_train = np.ones(len(train_inliers))

  new_dataset_tuples = ((x_train, y_train), (x_val, y_val), (
    x_test, y_test))
  utils.save_pickle(
      new_dataset_tuples,
      os.path.join(save_folder, 'v5_big_ztf_dataset_tuples.pkl'))
