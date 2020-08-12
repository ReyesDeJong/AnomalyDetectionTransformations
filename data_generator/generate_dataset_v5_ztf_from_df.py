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
  data_name = 'training_set_May-06-2020.pkl'
  data_folder = "/home/ereyes/Projects/Alerce/pickles"
  save_folder = '/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/'
  utils.check_path(save_folder)
  data_path = os.path.join(data_folder, data_name)

  n_classes = 5
  params = {
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.BATCH_SIZE: None,
    param_keys.N_INPUT_CHANNELS: 3,
    param_keys.CHANNELS_TO_USE: [0, 1, 2],
    param_keys.NANS_TO: 0,
    param_keys.CROP_SIZE: 21,
    param_keys.TEST_SIZE: n_classes * 200,
    param_keys.VAL_SIZE: n_classes * 100,
    param_keys.VALIDATION_RANDOM_SEED: random_seed,
    param_keys.CONVERTED_DATA_SAVEPATH: '/home/ereyes/Projects/Thesis/'
                                        'datasets/ALeRCE_data/ztf_v5/'
                                        'converted_data.pkl',
    param_keys.BOGUS_LABEL_VALUE: None,
  }

  # normal_data = get_df_dataset_from_name(params, data_path)
  frame_to_input = FrameToInput(params)
  frame_to_input.dataset_preprocessor.set_pipeline(
      [frame_to_input.dataset_preprocessor.check_single_image,
       frame_to_input.dataset_preprocessor.clean_misshaped,
       frame_to_input.dataset_preprocessor.select_channels,
       frame_to_input.dataset_preprocessor.crop_at_center,
       frame_to_input.dataset_preprocessor.normalize_by_image,
       frame_to_input.dataset_preprocessor.nan_to_num,
       ])
  single_dataset = frame_to_input.get_single_dataset()
  print('all data Set: ', np.unique(single_dataset.data_label,
                                    return_counts=True))

  stamps = single_dataset.data_array
  print(np.mean(np.max(stamps, axis=(1,2)), axis=0))
  print(np.mean(np.min(stamps, axis=(1, 2)), axis=0))

  labels = single_dataset.data_label
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
      os.path.join(save_folder, 'v5_big_ztf_dataset_tuples_new.pkl'))
