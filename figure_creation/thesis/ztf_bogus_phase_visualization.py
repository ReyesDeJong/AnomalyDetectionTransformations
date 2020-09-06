"""
Visualization of HiTS sample for paper, 4 samples per class
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

import matplotlib.pyplot as plt
from parameters import param_keys
from modules.data_loaders.frame_to_input import FrameToInput
import numpy as np
from modules import utils
from figure_creation.thesis.ztf_sample_visualization import plot_ztf_image
from modules.data_set_generic import Dataset


def get_dataset(path) -> Dataset:
    params = {
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.BATCH_SIZE: None,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.NANS_TO: 0,
        param_keys.CROP_SIZE: 63,
        param_keys.TEST_SIZE: 200,
        param_keys.VAL_SIZE: 100,
        param_keys.VALIDATION_RANDOM_SEED: RANDOM_SEED,
        param_keys.CONVERTED_DATA_SAVEPATH: '/home/ereyes/Projects/Thesis/'
                                            'datasets/ALeRCE_data/ztf_v5/'
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
    dataset = frame_to_input.get_single_dataset()
    return dataset


if __name__ == '__main__':
    SHOW = True
    N_SAMPLES_TO_PLOT = 5
    RANDOM_SEED = 234
    SAVE_FOLDER_NAME = 'bogus_phase_ztf'

    # data loader
    data_name = 'training_set_Aug-07-2020.pkl'
    data_folder = "/home/ereyes/Projects/Thesis/stamp_classifier_updater/data/"
    data_path = os.path.join(data_folder, data_name)
    # getting stamp_clf data
    dataset = get_dataset(data_path)
    dataset.shuffle_data(RANDOM_SEED)
    data_array = dataset.data_array
    data_labels = dataset.data_label
    print(dataset.data_array.shape)
    print('Data values per channel Min %s Max %s  Mean %s' % (
        np.mean(np.min(data_array, axis=(1, 2)), axis=0),
        np.mean(np.max(data_array, axis=(1, 2)), axis=0),
        np.mean(np.mean(data_array, axis=(1, 2)), axis=0)))
    print(np.unique(dataset.data_label, return_counts=True))
    # get Outliers
    for i in range(N_SAMPLES_TO_PLOT):
        plot_ztf_image(dataset.data_array[dataset.data_label == 4][i],
                       show=SHOW, name='outlier_%i' % i, plot_titles=not i,
                       save_folder_name=SAVE_FOLDER_NAME,
                       n_channels_to_plot=data_array.shape[-1])

    # plot_hits_many_images(dataset.data_array[dataset.data_label == 1][:4], show=True,
    #                name='aux')
