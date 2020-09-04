"""
Visualization of ztf sample for paper.

Should i undersample ZTF data to keep more SNe in small train? because large
dataset produce different preferences to different transformations

Small test set first appearances is for stamp clf data

label 5 to ashish boguses

class 1 for inliers and 0 for outliers
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
            images_of_label = dataset.data_array[dataset.data_label == label_i]
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
    SHOW = False
    LARGE_TRAIN_SET_PERCENTAGE = 0.9
    N_SAMPLES_TO_PLOT_PER_LABEL = 1
    RANDOM_SEED = 42
    SAVE_FOLDER_PATH = os.path.join(
        PROJECT_PATH, '..', 'datasets', 'thesis_data')
    outlier_original_label_value = 4

    # data loader
    train_set, val_set, test_set = load_ztf_stamp_clf_datasets()
    bogus_ashish_set = load_ashish_remaining_dataset()
    bogus_ashish_set.shuffle_data(random_seed=RANDOM_SEED)
    # label 5 to ashish bogus
    bogus_ashish_set.data_label = bogus_ashish_set.data_label + 5
    print('\nDisplay stamp clf data')
    display_dataset(train_set, N_SAMPLES_TO_PLOT_PER_LABEL, SHOW, 'train_set')
    display_dataset(val_set, N_SAMPLES_TO_PLOT_PER_LABEL, SHOW, 'val_set')
    display_dataset(test_set, N_SAMPLES_TO_PLOT_PER_LABEL, SHOW, 'test_set')
    display_dataset(bogus_ashish_set, N_SAMPLES_TO_PLOT_PER_LABEL*3, SHOW, 'ashish_bogus_set')
    del test_set

    # separate inliers-outliers
    print('\nseparate inliers-outliers')
    merged_dataset = Dataset(train_set.data_array, train_set.data_label, meta_data=train_set.meta_data)
    merged_dataset.append_dataset(val_set)
    # merged_dataset.undersample_data(20000, random_seed=RANDOM_SEED)
    merged_dataset.shuffle_data(RANDOM_SEED)
    display_dataset(merged_dataset, N_SAMPLES_TO_PLOT_PER_LABEL, SHOW,
                    'merged_set')
    data_array = merged_dataset.data_array
    data_labels = merged_dataset.data_label
    inlier_stamps = data_array[data_labels != outlier_original_label_value]
    inlier_labels = data_labels[data_labels != outlier_original_label_value]
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(inlier_stamps, show=SHOW, plot_titles=True,
                       title='inlier')
    outlier_stamps = data_array[data_labels == outlier_original_label_value]
    outlier_labels = data_labels[data_labels == outlier_original_label_value]
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(outlier_stamps, show=SHOW, plot_titles=True,
                       title='outlier')

    # small sets
    # SHOW = True
    print('\nsmall sets')
    small_train_n_samples = 7000
    small_val_n_samples = 1000
    small_test_n_samples = 3000
    inlier_indexes = np.arange(len(inlier_labels))
    print('starting_small_inlier ',
          len(inlier_indexes))
    # train
    small_train_indexes = inlier_indexes[:small_train_n_samples]
    small_inlier_remaining_indexes = inlier_indexes[small_train_n_samples:]
    print('small_inlier_remaining_indexes ',
          len(small_inlier_remaining_indexes))
    small_x_train = inlier_stamps[small_train_indexes]
    small_y_train = inlier_labels[small_train_indexes]
    print('small_y_train ', np.unique(small_y_train, return_counts=True))
    print(
        np.unique(small_y_train, return_counts=True)[1] / small_train_n_samples)
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(small_x_train, show=SHOW, plot_titles=True,
                       title='small xtrain')
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
        plot_ztf_image(small_x_val, show=SHOW, plot_titles=True,
                       title='small xval')
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
        np.unique(small_inliers_y_test, return_counts=True)[
            1] / small_test_n_samples)
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(small_inliers_x_test, show=SHOW, plot_titles=True,
                       title='small xtest inliers')
    # outliers
    # ashish outliers
    ashish_outlier_stamps = bogus_ashish_set.data_array
    ashish_outlier_labels = bogus_ashish_set.data_label
    ashish_outlier_indexes = np.arange(len(ashish_outlier_labels))
    print('starting ashish_outlier_indexes ',
          len(ashish_outlier_indexes))
    ashish_small_test_n_samples = int(len(ashish_outlier_labels)/2)
    ashish_small_test_indexes = ashish_outlier_indexes[
                                :ashish_small_test_n_samples]
    ashish_small_outlier_remaining_indexes = ashish_outlier_indexes[
                                             ashish_small_test_n_samples:]
    print('ashish_small_outlier_remaining_indexes ',
          len(ashish_small_outlier_remaining_indexes))
    ashish_small_outliers_x_test = ashish_outlier_stamps[
        ashish_small_test_indexes]
    ashish_small_outliers_y_test = ashish_outlier_labels[
        ashish_small_test_indexes]
    print('ashish_small_outliers_y_test ',
          np.unique(ashish_small_outliers_y_test, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(ashish_small_outliers_x_test, show=SHOW,
                       plot_titles=True,
                       title='ashish_small xtest outliers')
    # alerce clf outliers
    # TODO: change small_outlier_x_test for alerce_small_outlier_x_test
    outlier_indexes = np.arange(len(outlier_labels))
    print('starting alerce_outlier_indexes ',
          len(outlier_indexes))
    small_test_n_samples = small_test_n_samples - ashish_small_test_n_samples
    small_test_indexes = outlier_indexes[:small_test_n_samples]
    small_outlier_remaining_indexes = outlier_indexes[small_test_n_samples:]
    print('alerce_small_outlier_remaining_indexes ',
          len(small_outlier_remaining_indexes))
    small_outliers_x_test = outlier_stamps[small_test_indexes]
    small_outliers_y_test = outlier_labels[small_test_indexes]
    print('alerce_small_outliers_y_test ',
          np.unique(small_outliers_y_test, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(small_outliers_x_test, show=SHOW, plot_titles=True,
                       title='alerce_small xtest outliers')
    # ashish_outliers-alerce_outliers
    small_outliers_x_test = np.concatenate(
        [small_outliers_x_test, ashish_small_outliers_x_test])
    small_outliers_y_test = np.concatenate(
        [small_outliers_y_test, ashish_small_outliers_y_test])
    print('definiteive small_outliers_y_test ',
          np.unique(small_outliers_y_test, return_counts=True))
    display_dataset(Dataset(small_outliers_x_test, small_outliers_y_test),
                    N_SAMPLES_TO_PLOT_PER_LABEL * 2, SHOW,
                    'small xtest outliers')
    # inliers-outliers
    small_x_test = np.concatenate([small_inliers_x_test, small_outliers_x_test])
    small_y_test = np.concatenate([small_inliers_y_test, small_outliers_y_test])
    small_y_test_01_outlier_labels = np.concatenate(
        [np.ones_like(small_inliers_y_test),
         np.zeros_like(small_outliers_y_test)])
    print('small_y_test ', np.unique(small_y_test, return_counts=True))
    print('small_y_test_01_outlier_labels ',
          np.unique(small_y_test_01_outlier_labels, return_counts=True))
    # SHOW = True
    display_dataset(Dataset(small_x_test, small_y_test),
                    N_SAMPLES_TO_PLOT_PER_LABEL, SHOW,
                    'small xtest')
    small_dataset_dict = {
        general_keys.TRAIN: {general_keys.IMAGES: small_x_train,
                             general_keys.LABELS: small_y_train,
                             general_keys.OUTLIER_LABELS: np.ones_like(
                                 small_y_train)},
        general_keys.VALIDATION: {general_keys.IMAGES: small_x_val,
                                  general_keys.LABELS: small_y_val,
                                  general_keys.OUTLIER_LABELS: np.ones_like(
                                      small_y_val)},
        general_keys.TEST: {general_keys.IMAGES: small_x_test,
                            general_keys.LABELS: small_y_test,
                            general_keys.OUTLIER_LABELS: np.ones_like(
                                small_y_test_01_outlier_labels)},
    }
    utils.save_pickle(small_dataset_dict, os.path.join(
        SAVE_FOLDER_PATH, 'ztf_small_dict.pkl'))

    # large sets
    print('\nlarge sets')
    # SHOW = True
    # test
    # outliers
    # alerce
    alerce_large_outlier_indexes = small_outlier_remaining_indexes
    print('initial_alerce_large_outlier_indexes ',
          len(alerce_large_outlier_indexes))
    alerce_large_outliers_x_test = outlier_stamps[alerce_large_outlier_indexes]
    alerce_large_outliers_y_test = outlier_labels[alerce_large_outlier_indexes]
    print('alerce_large_outliers_y_test ',
          np.unique(alerce_large_outliers_y_test, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(alerce_large_outliers_x_test, show=SHOW, plot_titles=True,
                        title='alerce_Large xtest outliers')
    #ashish
    ashish_large_outlier_indexes = ashish_small_outlier_remaining_indexes
    print('initial_ashish_large_outlier_indexes ',
          len(ashish_large_outlier_indexes))
    ashish_large_outliers_x_test = ashish_outlier_stamps[ashish_large_outlier_indexes]
    ashish_large_outliers_y_test = ashish_outlier_labels[ashish_large_outlier_indexes]
    print('ashish_large_outliers_y_test ',
          np.unique(ashish_large_outliers_y_test, return_counts=True))
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(ashish_large_outliers_x_test, show=SHOW, plot_titles=True,
                        title='ashish_Large xtest outliers')
    # ashish_outliers-alerce_outliers
    large_outliers_x_test = np.concatenate(
        [alerce_large_outliers_x_test, ashish_large_outliers_x_test])
    large_outliers_y_test = np.concatenate(
        [alerce_large_outliers_y_test, ashish_large_outliers_y_test])
    print('definiteive large_outliers_y_test ',
          np.unique(large_outliers_y_test, return_counts=True))
    display_dataset(Dataset(large_outliers_x_test, large_outliers_y_test),
                    N_SAMPLES_TO_PLOT_PER_LABEL * 2, SHOW,
                    'large xtest outliers')
    # inliers
    large_inlier_indexes = small_inlier_remaining_indexes
    print('initial_large_inlier_indexes ',
          len(large_inlier_indexes))
    large_test_inliers_n_samples = len(large_outliers_y_test)
    large_inlier_test_indexes = large_inlier_indexes[
                                     :large_test_inliers_n_samples]
    large_inlier_remaining_indexes = large_inlier_indexes[
                                     large_test_inliers_n_samples:]
    print('large_inlier_remaining_indexes ',
          len(large_inlier_remaining_indexes))
    large_inliers_x_test = inlier_stamps[large_inlier_test_indexes]
    large_inliers_y_test = inlier_labels[large_inlier_test_indexes]
    print('large_inliers_y_test ',
          np.unique(large_inliers_y_test, return_counts=True))
    print(
        np.unique(large_inliers_y_test, return_counts=True)[1] / large_test_inliers_n_samples)
    for i in range(N_SAMPLES_TO_PLOT_PER_LABEL):
        plot_ztf_image(large_inliers_x_test, show=SHOW, plot_titles=True,
                        title='Large xtest inliers')
    # inliers-outliers
    large_x_test = np.concatenate([large_inliers_x_test, large_outliers_x_test])
    large_y_test = np.concatenate([large_inliers_y_test, large_outliers_y_test])
    large_y_test_01_outlier_labels = np.concatenate(
        [np.ones_like(large_inliers_y_test),
         np.zeros_like(large_outliers_y_test)])
    print('large_y_test ', np.unique(large_y_test, return_counts=True))
    print('large_y_test_01_outlier_labels ',
          np.unique(large_y_test_01_outlier_labels, return_counts=True))
    # SHOW = True
    display_dataset(Dataset(large_x_test, large_y_test),
                    N_SAMPLES_TO_PLOT_PER_LABEL, SHOW, 'large xtest')
    # train
    #####
    # include of small train and vall inlier indexes
    #####
    large_inlier_remaining_indexes = np.concatenate([large_inlier_remaining_indexes, small_val_indexes, small_train_indexes])
    np.random.RandomState(RANDOM_SEED).shuffle(large_inlier_remaining_indexes)
    print('train-val_large_inlier_indexes ',
          len(large_inlier_remaining_indexes))
    large_train_n_samples = int(np.round(len(large_inlier_remaining_indexes) * 0.9))
    large_train_indexes = large_inlier_remaining_indexes[:large_train_n_samples]
    large_inlier_remaining_indexes = large_inlier_remaining_indexes[large_train_n_samples:]
    print('large_inlier_remaining_indexes ',
          len(large_inlier_remaining_indexes))
    large_x_train = inlier_stamps[large_train_indexes]
    large_y_train = inlier_labels[large_train_indexes]
    print('large_y_train ', np.unique(large_y_train, return_counts=True), large_x_train.shape)
    print(
        np.unique(large_y_train, return_counts=True)[1] / large_train_n_samples)
    display_dataset(Dataset(large_x_train, large_y_train),
                    N_SAMPLES_TO_PLOT_PER_LABEL, show=SHOW,
                    set_name='large_train')
    # val
    large_val_indexes = large_inlier_remaining_indexes
    print('large_inlier_remaining_indexes ',
          len(large_inlier_remaining_indexes))
    large_x_val = inlier_stamps[large_val_indexes]
    large_y_val = inlier_labels[large_val_indexes]
    print('large_y_val ', np.unique(large_y_val, return_counts=True), large_x_val.shape)
    print(
        np.unique(large_y_val, return_counts=True)[1] / len(large_val_indexes))
    display_dataset(Dataset(large_x_val, large_y_val), N_SAMPLES_TO_PLOT_PER_LABEL, show=SHOW, set_name='large_val')
    large_dataset_dict = {
        general_keys.TRAIN: {general_keys.IMAGES: large_x_train,
                             general_keys.LABELS: large_y_train,
                             general_keys.OUTLIER_LABELS: np.ones_like(
                                 large_y_train)},
        general_keys.VALIDATION: {general_keys.IMAGES: large_x_val,
                                  general_keys.LABELS: large_y_val,
                                  general_keys.OUTLIER_LABELS: np.ones_like(
                                      large_y_val)},
        general_keys.TEST: {general_keys.IMAGES: large_x_test,
                            general_keys.LABELS: large_y_test,
                            general_keys.OUTLIER_LABELS: np.ones_like(
                                large_y_test_01_outlier_labels)},
    }
    utils.save_pickle(large_dataset_dict, os.path.join(
        SAVE_FOLDER_PATH, 'ztf_large_dict.pkl'))
