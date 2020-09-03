"""
Visualization of ztf sample for paper.

Should i undersample ZTF data to keep more SNe in small train? because large
dataset produce different preferences to different transformations
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from parameters import general_keys
import numpy as np
from figure_creation.thesis.ztf_sample_visualization import plot_ztf_image
from modules import utils
from typing import List
import pandas as pd
from modules.data_set_generic import Dataset


def dataset_from_dict(data_dict, set_key) -> Dataset:
    return Dataset(data_dict[set_key][general_keys.IMAGES],
                   data_dict[set_key][general_keys.LABELS])


def load_ztf_stamp_clf_datasets() -> List[Dataset]:
    data_path = os.path.join(
        PROJECT_PATH, '..', 'datasets', 'thesis_data', 'ztfv7_stamp_clf_data',
        'ztfv7_stamp_clf_processed.pkl')
    data_dict = pd.read_pickle(data_path)
    train_set = dataset_from_dict(data_dict, 'Train')
    val_set = dataset_from_dict(data_dict, 'Validation')
    test_set = dataset_from_dict(data_dict, 'Test')
    return train_set, val_set, test_set


def load_ashish_remaining_dataset() -> Dataset:
    data_path = os.path.join(
        PROJECT_PATH, '..', 'datasets', 'thesis_data', 'ztfv7_stamp_clf_data',
        'ashish_bogus_remaining_processed.pkl')
    data_dict = pd.read_pickle(data_path)
    return Dataset(data_dict[general_keys.IMAGES],
                   data_dict[general_keys.LABELS])


def plot_n_samples_per_class(dataset: Dataset, n_samples_per_class, show=False,
    set_name=''):
    label_values = np.unique(dataset.data_label)
    for label_i in label_values:
        for i in range(n_samples_per_class):
            images_of_label = dataset.data_array[dataset.data_label==label_i]
            sample_index = np.random.randint(len(images_of_label))
            plot_ztf_image(images_of_label[sample_index], show=show,
                           plot_titles=True, title='%s_%i_class_%i' % (
                set_name, sample_index, label_i))

def display_dataset(dataset: Dataset, n_samples_per_class, show=False,
    set_name=''):
    print(set_name)
    print(dataset.data_label[-10:])
    print(dataset.data_array.shape)
    print('Data values per channel Min %s Max %s  Mean %s' % (
        np.mean(np.min(dataset.data_array, axis=(1, 2)), axis=0),
        np.mean(np.max(dataset.data_array, axis=(1, 2)), axis=0),
        np.mean(np.mean(dataset.data_array, axis=(1, 2)), axis=0)))
    print(np.unique(dataset.data_label, return_counts=True))
    plot_n_samples_per_class(dataset, n_samples_per_class, show, set_name)


if __name__ == '__main__':
    SHOW = True
    N_SAMPLES_TO_PLOT_PER_LABEL = 1
    RANDOM_SEED = 42
    SAVE_FOLDER_PATH = os.path.join(
        PROJECT_PATH, '..', 'datasets', 'thesis_data')
    outlier_original_label_value = 4

    # data loader
    train_set, val_set, test_set = load_ztf_stamp_clf_datasets()
    bogus_ashish_set = load_ashish_remaining_dataset()
    print('\nDisplay stamp clf data')
    # display_dataset(train_set, N_SAMPLES_TO_PLOT_PER_LABEL, SHOW, 'train_set')
    # display_dataset(val_set, N_SAMPLES_TO_PLOT_PER_LABEL, SHOW, 'val_set')
    # display_dataset(test_set, N_SAMPLES_TO_PLOT_PER_LABEL, SHOW, 'test_set')
    # display_dataset(bogus_ashish_set, N_SAMPLES_TO_PLOT_PER_LABEL*3, SHOW, 'ashish_bogus_set')
    del test_set

    # separate inliers-outliers
    print('\nseparate inliers-outliers')
    merged_dataset = train_set
    merged_dataset.append_dataset(val_set)
    # merged_dataset.undersample_data(20000, random_seed=RANDOM_SEED)
    merged_dataset.shuffle_data(RANDOM_SEED)
    display_dataset(merged_dataset, N_SAMPLES_TO_PLOT_PER_LABEL, SHOW, 'merged_set')
    data_array = merged_dataset.data_array
    data_labels = merged_dataset.data_label
    inlier_stamps = data_array[data_labels != outlier_original_label_value]
    inlier_labels = data_labels[data_labels != outlier_original_label_value]
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(inlier_stamps[-i], show=SHOW, plot_titles=True,
                       title='inlier')
    outlier_stamps = data_array[data_labels == outlier_original_label_value]
    outlier_labels = data_labels[data_labels == outlier_original_label_value]
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(outlier_stamps[-i], show=SHOW, plot_titles=True,
                        title='outlier')

    # small sets
    # SHOW = True
    print('\nsmall sets')
    small_train_n_samples = 7000
    small_val_n_samples = 1000
    small_test_n_samples = 3000
    inlier_indexes = np.arange(len(inlier_labels))
    print('starting_small_inlier_remaining ',
          len(inlier_indexes))
    # train
    small_train_indexes = inlier_indexes[:small_train_n_samples]
    small_inlier_remaining_indexes = inlier_indexes[small_train_n_samples:]
    print('small_inlier_remaining_indexes ',
          len(small_inlier_remaining_indexes))
    small_x_train = inlier_stamps[small_train_indexes]
    small_y_train = inlier_labels[small_train_indexes]
    print('small_y_train ', np.unique(small_y_train, return_counts=True))
    print(np.unique(small_y_train, return_counts=True)[1]/small_train_n_samples)
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(small_x_train[-i], show=SHOW, plot_titles=True, title='small xtrain')
    # val
    small_val_indexes = small_inlier_remaining_indexes[:small_val_n_samples]
    small_inlier_remaining_indexes = small_inlier_remaining_indexes[
                                     small_val_n_samples:]
    print('small_inlier_remaining_indexes ',
          len(small_inlier_remaining_indexes))
    small_x_val = inlier_stamps[small_val_indexes]
    small_y_val = inlier_labels[small_val_indexes]
    print('small_y_val ', np.unique(small_y_val, return_counts=True))
    print(
        np.unique(small_y_val, return_counts=True)[1] / small_val_n_samples)
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(small_x_val[-i], show=SHOW, plot_titles=True, title='small xval')
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
    print(
        np.unique(small_inliers_y_test, return_counts=True)[1] / small_test_n_samples)
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(small_inliers_x_test[-i], show=SHOW, plot_titles=True,
                                                title='small xtest inliers')
    # # outliers
    # outlier_indexes = large_outlier_remaining_indexes
    # print('starting small_outlier_remaining_indexes ',
    #       len(outlier_indexes))
    # small_test_indexes = outlier_indexes[:small_test_n_samples]
    # small_outlier_remaining_indexes = outlier_indexes[small_test_n_samples:]
    # print('small_outlier_remaining_indexes ',
    #       len(small_outlier_remaining_indexes))
    # small_outliers_x_test = outlier_stamps[small_test_indexes]
    # small_outliers_y_test = outlier_labels[small_test_indexes]
    # print('small_outliers_y_test ',
    #       np.unique(small_outliers_y_test, return_counts=True))
    # for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
    #     plot_ztf_image(small_outliers_x_test[-i], show=SHOW, plot_titles=True,
    #                     n_channels_to_plot=n_channels,
    #                     title='small xtest outliers')
    # # inliers-outliers
    # small_x_test = np.concatenate([small_inliers_x_test, small_outliers_x_test])
    # small_y_test = np.concatenate([small_inliers_y_test, small_outliers_y_test])
    # print('small_y_test ', np.unique(small_y_test, return_counts=True))
    # # SHOW = True
    # for i in range(N_SAMPLES_TO_PLOT_PER_LABEL * 2):
    #     sample_i = np.random.randint(len(small_x_test))
    #     plot_ztf_image(small_x_test[-sample_i], show=SHOW, plot_titles=True,
    #                     n_channels_to_plot=n_channels,
    #                     title='small xtest %i' % small_y_test[-sample_i])
    # small_data_tuples = (
    #     (small_x_train, small_y_train), (small_x_val, small_y_val), (
    #         small_x_test, small_y_test))
    # utils.save_pickle(small_data_tuples, os.path.join(
    #     SAVE_FOLDER_PATH, 'ztf_small_%ic_tuples.pkl' % n_channels))

    # # large sets
    # print('\nlarge sets')
    # large_train_n_samples = 70000
    # large_val_n_samples = 10000
    # large_test_n_samples = 30000
    # inlier_indexes = np.arange(len(inlier_labels))
    # # train
    # large_train_indexes = inlier_indexes[:large_train_n_samples]
    # large_inlier_remaining_indexes = inlier_indexes[large_train_n_samples:]
    # print('large_inlier_remaining_indexes ',
    #       len(large_inlier_remaining_indexes))
    # large_x_train = inlier_stamps[large_train_indexes]
    # large_y_train = inlier_labels[large_train_indexes]
    # print('large_y_train ', np.unique(large_y_train, return_counts=True))
    # for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
    #     plot_ztf_image(large_x_train[-i], show=SHOW, plot_titles=True,
    #                     n_channels_to_plot=n_channels, title='Large xtrain')
    # # val
    # large_val_indexes = large_inlier_remaining_indexes[:large_val_n_samples]
    # large_inlier_remaining_indexes = large_inlier_remaining_indexes[
    #                                  large_val_n_samples:]
    # print('large_inlier_remaining_indexes ',
    #       len(large_inlier_remaining_indexes))
    # large_x_val = inlier_stamps[large_val_indexes]
    # large_y_val = inlier_labels[large_val_indexes]
    # print('large_y_val ', np.unique(large_y_val, return_counts=True))
    # for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
    #     plot_ztf_image(large_x_val[-i], show=SHOW, plot_titles=True,
    #                     n_channels_to_plot=n_channels, title='Large xval')
    # # test
    # large_test_indexes = large_inlier_remaining_indexes[:large_test_n_samples]
    # large_inlier_remaining_indexes = large_inlier_remaining_indexes[
    #                                  large_test_n_samples:]
    # print('large_inlier_remaining_indexes ',
    #       len(large_inlier_remaining_indexes))
    # large_inliers_x_test = inlier_stamps[large_test_indexes]
    # large_inliers_y_test = inlier_labels[large_test_indexes]
    # print('large_inliers_y_test ',
    #       np.unique(large_inliers_y_test, return_counts=True))
    # for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
    #     plot_ztf_image(large_inliers_x_test[-i], show=SHOW, plot_titles=True,
    #                     n_channels_to_plot=n_channels,
    #                     title='Large xtest inliers')
    # # outliers
    # outlier_indexes = np.arange(len(outlier_labels))
    # large_test_indexes = outlier_indexes[:large_test_n_samples]
    # large_outlier_remaining_indexes = outlier_indexes[large_test_n_samples:]
    # print('large_outlier_remaining_indexes ',
    #       len(large_outlier_remaining_indexes))
    # large_outliers_x_test = outlier_stamps[large_test_indexes]
    # large_outliers_y_test = outlier_labels[large_test_indexes]
    # print('large_outliers_y_test ',
    #       np.unique(large_outliers_y_test, return_counts=True))
    # for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
    #     plot_ztf_image(large_outliers_x_test[-i], show=SHOW, plot_titles=True,
    #                     n_channels_to_plot=n_channels,
    #                     title='Large xtest outliers')
    # # inliers-outliers
    # large_x_test = np.concatenate([large_inliers_x_test, large_outliers_x_test])
    # large_y_test = np.concatenate([large_inliers_y_test, large_outliers_y_test])
    # print('large_y_test ', np.unique(large_y_test, return_counts=True))
    # # SHOW = True
    # for i in range(N_SAMPLES_TO_PLOT_PER_LABEL * 2):
    #     sample_i = np.random.randint(len(large_x_test))
    #     plot_ztf_image(large_x_test[-sample_i], show=SHOW, plot_titles=True,
    #                     n_channels_to_plot=n_channels,
    #                     title='Large xtest %i' % large_y_test[-sample_i])
    # large_data_tuples = (
    #     (large_x_train, large_y_train), (large_x_val, large_y_val), (
    #         large_x_test, large_y_test))
    # utils.save_pickle(large_data_tuples, os.path.join(
    #     SAVE_FOLDER_PATH, 'ztf_large_%ic_tuples.pkl' % n_channels))
    #

