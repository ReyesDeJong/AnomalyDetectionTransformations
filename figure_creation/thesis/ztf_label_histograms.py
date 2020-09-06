"""
Visualization of HiTS sample for paper, 4 samples per class
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

import matplotlib.pyplot as plt
from parameters import param_keys, general_keys
import pandas as pd
from modules.data_loaders.frame_to_input import FrameToInput
import numpy as np
from modules import utils
from data_generator.thesis_ztf import load_ztf_stamp_clf_datasets, \
    display_dataset
from modules.data_set_generic import Dataset

if __name__ == '__main__':
    SHOW = False
    SAVE_FIG = True
    DPI = 300
    ALPHA = 0.3
    FIG_FORMAT = 'pdf'
    FOLDER_SAVE_NAME = 'ztf_histograms'

    # data loading
    folder_save_root_path = os.path.join(
        PROJECT_PATH, 'figure_creation/thesis/figs', FOLDER_SAVE_NAME)
    utils.check_path(folder_save_root_path)
    # stmp_clf_data
    train_set, val_set, test_set = load_ztf_stamp_clf_datasets()
    # display_dataset(train_set, 0, False, 'train')
    all_dataset = Dataset(train_set.data_array, train_set.data_label)
    all_dataset.append_dataset(val_set).append_dataset(test_set)
    # display_dataset(all_dataset, 0, False, 'all data dataset')
    outlier_detection_dataset = Dataset(train_set.data_array, train_set.data_label)
    outlier_detection_dataset.append_dataset(val_set)
    # display_dataset(outlier_detection_dataset, 0, False, 'outlier_detection_dataset')
    # outlier_det_data
    outlier_detection_data_folder_path = os.path.join(
        PROJECT_PATH, '..', 'datasets', 'thesis_data')
    small_file_name = 'ztf_small_dict.pkl'
    large_file_name = 'ztf_large_dict.pkl'
    small_data_dict = pd.read_pickle(os.path.join(outlier_detection_data_folder_path, small_file_name))
    large_data_dict = pd.read_pickle(os.path.join(outlier_detection_data_folder_path, large_file_name))

    # plot all labels hist
    fig_name = 'ztf_all_samples_label_hist'
    x = ['AGN', 'SN', 'VS', 'Asteroid', 'Bogus']
    y = np.unique(all_dataset.data_label, return_counts=True)[1]
    fig, ax = plt.subplots()
    width = 0.75  # the width of the bars
    ind = np.arange(len(x))  # the x locations for the groups
    rects = ax.bar(ind, y, width)
    ax.set_xticks(ind)
    ax.set_xticklabels(x, minor=False)
    for i, v in enumerate(y):
        ax.text(i, v + np.max(y) * 0.005, str(v), color='black', ha='center')
    plt.xlabel('Label names')
    plt.ylabel('Sample count')
    plt.grid(axis='y', linestyle='--')
    if SHOW:
        plt.show()
    if SAVE_FIG:
        plt.savefig(
            os.path.join(
                folder_save_root_path,
                '%s.%s' % (fig_name, FIG_FORMAT)), dpi=DPI, format=FIG_FORMAT,
            bbox_inches='tight', pad_inches=0, transparent=False)

    # plot inliers hist
    fig_name = 'ztf_inliers_label_hist'
    fig, ax = plt.subplots()
    width = 0.75  # the width of the bars
    x = ['AGN', 'SN', 'VS', 'Asteroid']
    ind = np.arange(len(x))  # the x locations for the groups
    set_version_name_list = ['Small', 'Large']
    set_dict_list = [small_data_dict, large_data_dict]
    set_name = [general_keys.TRAIN, general_keys.VALIDATION, general_keys.TEST]
    for set_dict_index, set_dict in enumerate(set_dict_list):
        for set_name_i in set_name:
            set_version_name = set_version_name_list[set_dict_index]
            ploted_set_name = '%s %s' % (set_version_name, set_name_i)
            images = set_dict[set_name_i][general_keys.IMAGES]
            outlier_det_labels = set_dict[set_name_i][general_keys.OUTLIER_LABELS]
            true_labels = set_dict[set_name_i][general_keys.LABELS]
            inlier_true_labels = true_labels[outlier_det_labels==1]
            #plot histogram
            print(ploted_set_name)
            y = np.unique(inlier_true_labels, return_counts=True)[1]
            print(y)
            y = y / np.sum(y)
            print(y)
            rects = ax.bar(ind, y, width, alpha=ALPHA, label=ploted_set_name)
            # ax.set_yscale('log')
    ax.set_xticks(ind)
    ax.set_xticklabels(x, minor=False)
    # plt.title('title')
    plt.xlabel('Label names')
    plt.ylabel('Outlier detection inliers sample percentage')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--')
    if SHOW:
        plt.show()
    if SAVE_FIG:
        plt.savefig(
            os.path.join(
                folder_save_root_path,
                '%s.%s' % (fig_name, FIG_FORMAT)), dpi=DPI, format=FIG_FORMAT,
            bbox_inches='tight', pad_inches=0, transparent=False)
