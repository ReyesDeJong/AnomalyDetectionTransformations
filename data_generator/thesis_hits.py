"""
Visualization of HiTS sample for paper, 4 samples per class
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from parameters import loader_keys, general_keys
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
import numpy as np
from figure_creation.thesis.hits_sample_visualization import plot_hits_image
from modules import utils

if __name__ == '__main__':
    CHANNELS_TO_USE = [0, 1, 2, 3]
    SHOW = False
    N_SAMPLES_TO_PLOT = 3
    RANDOM_SEED = 42
    SAVE_FOLDER_PATH = os.path.join(
        PROJECT_PATH, '..', 'datasets', 'thesis_data', 'hits')
    utils.check_path(SAVE_FOLDER_PATH)
    n_channels = len(CHANNELS_TO_USE)
    outlier_original_label_value = 0
    # data loader
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 150000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: CHANNELS_TO_USE,
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: RANDOM_SEED,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    hits_loader = HiTSOutlierLoader(hits_params)
    hits_loader.set_pickle_loading(False)
    hits_loader.set_pickle_saving(False)

    # getting data
    dataset = hits_loader.get_preprocessed_unsplitted_dataset()
    print(dataset.data_label)
    dataset.shuffle_data(RANDOM_SEED)
    data_array = dataset.data_array
    data_labels = dataset.data_label
    print(dataset.data_label)
    print(dataset.data_array.shape)
    print('Data values per channel Min %s Max %s  Mean %s' % (
        np.mean(np.min(data_array, axis=(1, 2)), axis=0),
        np.mean(np.max(data_array, axis=(1, 2)), axis=0),
        np.mean(np.mean(data_array, axis=(1, 2)), axis=0)))
    print(np.unique(dataset.data_label, return_counts=True))

    # separate inliers-outliers
    print('\nseparate inliers-outliers')
    inlier_stamps = data_array[data_labels != outlier_original_label_value]
    inlier_labels = data_labels[data_labels != outlier_original_label_value]
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(inlier_stamps[-i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels, title='inlier')
    outlier_stamps = data_array[data_labels == outlier_original_label_value]
    outlier_labels = data_labels[data_labels == outlier_original_label_value]
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(outlier_stamps[-i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels, title='outlier')

    # large sets
    print('\n---large sets')
    large_train_n_samples = 70000
    large_val_n_samples = 10000
    large_test_n_samples = 30000
    inlier_indexes = np.arange(len(inlier_labels))
    # train
    large_train_indexes = inlier_indexes[:large_train_n_samples]
    large_inlier_remaining_indexes = inlier_indexes[large_train_n_samples:]
    print('large_inlier_remaining_indexes ',
          len(large_inlier_remaining_indexes))
    large_x_train = inlier_stamps[large_train_indexes]
    large_y_train = inlier_labels[large_train_indexes]
    print('large_y_train ', np.unique(large_y_train, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(large_x_train[-i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels, title='Large xtrain')
    # val
    large_val_indexes = large_inlier_remaining_indexes[:large_val_n_samples]
    large_inlier_remaining_indexes = large_inlier_remaining_indexes[
                                     large_val_n_samples:]
    print('large_inlier_remaining_indexes ',
          len(large_inlier_remaining_indexes))
    large_x_val = inlier_stamps[large_val_indexes]
    large_y_val = inlier_labels[large_val_indexes]
    print('large_y_val ', np.unique(large_y_val, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(large_x_val[-i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels, title='Large xval')
    # test
    large_test_indexes = large_inlier_remaining_indexes[:large_test_n_samples]
    large_inlier_remaining_indexes = large_inlier_remaining_indexes[
                                     large_test_n_samples:]
    print('large_inlier_remaining_indexes ',
          len(large_inlier_remaining_indexes))
    large_inliers_x_test = inlier_stamps[large_test_indexes]
    large_inliers_y_test = inlier_labels[large_test_indexes]
    print('large_inliers_y_test ',
          np.unique(large_inliers_y_test, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(large_inliers_x_test[-i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels,
                        title='Large xtest inliers')
    # outliers
    outlier_indexes = np.arange(len(outlier_labels))
    large_test_indexes = outlier_indexes[:large_test_n_samples]
    large_outlier_remaining_indexes = outlier_indexes[large_test_n_samples:]
    print('large_outlier_remaining_indexes ',
          len(large_outlier_remaining_indexes))
    large_outliers_x_test = outlier_stamps[large_test_indexes]
    large_outliers_y_test = outlier_labels[large_test_indexes]
    print('large_outliers_y_test ',
          np.unique(large_outliers_y_test, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(large_outliers_x_test[-i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels,
                        title='Large xtest outliers')
    # inliers-outliers
    large_x_test = np.concatenate([large_inliers_x_test, large_outliers_x_test])
    large_y_test = np.concatenate([large_inliers_y_test, large_outliers_y_test])
    print('large_y_test ', np.unique(large_y_test, return_counts=True))
    # SHOW = True
    for i in range(N_SAMPLES_TO_PLOT * 2):
        sample_i = np.random.randint(len(large_x_test))
        plot_hits_image(large_x_test[-sample_i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels,
                        title='Large xtest %i' % large_y_test[-sample_i])
    large_data_tuples = (
        (large_x_train, large_y_train), (large_x_val, large_y_val), (
            large_x_test, large_y_test))
    utils.save_pickle(large_data_tuples, os.path.join(
        SAVE_FOLDER_PATH, 'hits_large_%ic_tuples.pkl' % n_channels))

    # small sets
    # SHOW = True
    print('\n---small sets')
    small_train_n_samples = 7000
    small_val_n_samples = 1000
    small_test_n_samples = 3000
    inlier_indexes = large_inlier_remaining_indexes
    print('starting small_inlier_remaining_indexes ',
          len(inlier_indexes))
    # train
    small_train_indexes = inlier_indexes[:small_train_n_samples]
    small_inlier_remaining_indexes = inlier_indexes[small_train_n_samples:]
    print('small_inlier_remaining_indexes ',
          len(small_inlier_remaining_indexes))
    small_x_train = inlier_stamps[small_train_indexes]
    small_y_train = inlier_labels[small_train_indexes]
    print('small_y_train ', np.unique(small_y_train, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(small_x_train[-i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels, title='small xtrain')
    # val
    small_val_indexes = small_inlier_remaining_indexes[:small_val_n_samples]
    small_inlier_remaining_indexes = small_inlier_remaining_indexes[
                                     small_val_n_samples:]
    print('small_inlier_remaining_indexes ',
          len(small_inlier_remaining_indexes))
    small_x_val = inlier_stamps[small_val_indexes]
    small_y_val = inlier_labels[small_val_indexes]
    print('small_y_val ', np.unique(small_y_val, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(small_x_val[-i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels, title='small xval')
    # test
    small_test_indexes = small_inlier_remaining_indexes[:small_test_n_samples]
    small_inlier_remaining_indexes = small_inlier_remaining_indexes[
                                     small_test_n_samples:]
    print('small_inlier_remaining_indexes ',
          len(small_inlier_remaining_indexes))
    small_inliers_x_test = inlier_stamps[small_test_indexes]
    small_inliers_y_test = inlier_labels[small_test_indexes]
    print('small_inliers_y_test ',
          np.unique(small_inliers_y_test, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(small_inliers_x_test[-i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels,
                        title='small xtest inliers')
    # outliers
    outlier_indexes = large_outlier_remaining_indexes
    print('starting small_outlier_remaining_indexes ',
          len(outlier_indexes))
    small_test_indexes = outlier_indexes[:small_test_n_samples]
    small_outlier_remaining_indexes = outlier_indexes[small_test_n_samples:]
    print('small_outlier_remaining_indexes ',
          len(small_outlier_remaining_indexes))
    small_outliers_x_test = outlier_stamps[small_test_indexes]
    small_outliers_y_test = outlier_labels[small_test_indexes]
    print('small_outliers_y_test ',
          np.unique(small_outliers_y_test, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(small_outliers_x_test[-i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels,
                        title='small xtest outliers')
    # inliers-outliers
    small_x_test = np.concatenate([small_inliers_x_test, small_outliers_x_test])
    small_y_test = np.concatenate([small_inliers_y_test, small_outliers_y_test])
    print('small_y_test ', np.unique(small_y_test, return_counts=True))
    # SHOW = True
    for i in range(N_SAMPLES_TO_PLOT * 2):
        sample_i = np.random.randint(len(small_x_test))
        plot_hits_image(small_x_test[-sample_i], show=SHOW, plot_titles=True,
                        n_channels_to_plot=n_channels,
                        title='small xtest %i' % small_y_test[-sample_i])
    small_data_tuples = (
        (small_x_train, small_y_train), (small_x_val, small_y_val), (
            small_x_test, small_y_test))
    utils.save_pickle(small_data_tuples, os.path.join(
        SAVE_FOLDER_PATH, 'hits_small_%ic_tuples.pkl' % n_channels))
