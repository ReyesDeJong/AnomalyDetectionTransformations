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
from data_generator.thesis_ztf import display_dataset


def get_dataset(path) -> Dataset:
    params = {
        param_keys.DATA_PATH_TRAIN: path,
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
    SHOW = False
    N_SAMPLES_TO_PLOT = 10
    RANDOM_SEED = 234
    SAVE_FOLDER_NAME = 'bogus_phase_ztf'

    # data loader
    data_name = 'training_set_Aug-07-2020.pkl'
    data_folder = "/home/ereyes/Projects/Thesis/stamp_classifier_updater/data/"
    alerce_data_path = os.path.join(data_folder, data_name)
    ashish_data_path = "/home/ereyes/Projects/Alerce/pickles/bogus_ashish.pkl"
    # getting stamp_clf data
    alerce_dataset = get_dataset(alerce_data_path)
    alerce_dataset.shuffle_data(RANDOM_SEED)
    ashish_dataset = get_dataset(ashish_data_path)
    ashish_dataset.shuffle_data(RANDOM_SEED)
    display_dataset(alerce_dataset, 0, False, 'alerce')
    display_dataset(ashish_dataset, 0, False, 'ashish')
    # get Outliers
    for i in range(N_SAMPLES_TO_PLOT):
        plot_ztf_image(alerce_dataset.data_array[alerce_dataset.data_label == 4][i],
                       show=SHOW, name='alerce_outlier_%i' % i, plot_titles=False,#not i,
                       save_folder_name=SAVE_FOLDER_NAME,
                       n_channels_to_plot=3)
        plot_ztf_image(ashish_dataset.data_array[i],
                       show=SHOW, name='ashish_outlier_%i' % i, plot_titles=False, #not i,
                       save_folder_name=SAVE_FOLDER_NAME,
                       n_channels_to_plot=3)
