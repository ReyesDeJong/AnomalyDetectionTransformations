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
  chunk_i = 1
  random_seed = 42
  val_inlier_percentage = 0.1
  data_name = 'unlabeled_set_images_%i.pkl' % chunk_i
  data_folder = "/home/ereyes/Projects/Thesis/stamp_classifier_updater/data/" \
                "last_n_days_alerts"
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

  utils.save_pickle(
      stamps,
      os.path.join(save_folder, 'unlabeled_ztf_chunk_%i.pkl' % chunk_i))
