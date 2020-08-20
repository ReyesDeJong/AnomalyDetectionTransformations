import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import pandas as pd
from parameters import param_keys
from modules.data_loaders.frame_to_input import FrameToInput
import numpy as np
from modules import utils

if __name__ == "__main__":
    n_samples = 20
    random_seed = 24
    val_inlier_percentage = 0.1
    data_name = 'training_set_Aug-07-2020.pkl'
    data_folder = "/home/ereyes/Projects/Thesis/stamp_classifier_updater/data/"
    save_folder = '/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/' \
                  'small_df_debug'
    utils.check_path(save_folder)
    save_path = os.path.join(save_folder, 'ztf_v7_n_samples_df.pkl')
    data_path = os.path.join(data_folder, data_name)

    ztf_v7_dataframe = pd.read_pickle(data_path)

    indexes_list = list(range(len(ztf_v7_dataframe)))
    np.random.RandomState(random_seed).shuffle(indexes_list)
    indexes_n_samples = indexes_list[: n_samples]
    ztf_v7_n_samples_df = ztf_v7_dataframe.iloc[indexes_n_samples].reset_index(
        drop=True)
    print(ztf_v7_n_samples_df)
    utils.save_pickle(ztf_v7_n_samples_df, save_path)

    n_classes = 5
    params = {
        param_keys.DATA_PATH_TRAIN: save_path,
        param_keys.BATCH_SIZE: None,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.NANS_TO: 0,
        param_keys.CROP_SIZE: 21,
        param_keys.TEST_SIZE: n_classes * 300,
        param_keys.VAL_SIZE: n_classes * 100,
        param_keys.VALIDATION_RANDOM_SEED: random_seed,
        param_keys.CONVERTED_DATA_SAVEPATH: '/home/ereyes/Projects/Thesis/'
                                            'datasets/ALeRCE_data/'
                                            'converted_data.pkl',
        param_keys.BOGUS_LABEL_VALUE: None,
    }

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
    utils.save_pickle(
        stamps, os.path.join(save_folder, 'ztf_v7_n_samples_stamps.pkl'))
