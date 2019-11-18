"""
ZTF stamps outlier loader
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.data_set_generic import Dataset
from parameters import general_keys, param_keys
from transformations import AbstractTransformer
from parameters import loader_keys
from modules.data_loaders.frame_to_input import FrameToInput
from modules import utils


class ZTFOutlierLoader(object):

    def __init__(self, params: dict, transformer: AbstractTransformer = None):
        self.data_path = params[loader_keys.DATA_PATH]
        self.val_inlier_percentage = params[loader_keys.VAL_SET_INLIER_PERCENTAGE]
        self.used_channels = params[loader_keys.USED_CHANNELS]
        self.crop_size = params[loader_keys.CROP_SIZE]
        self.random_seed = params[general_keys.RANDOM_SEED]
        self.converted_data_path = self._get_converted_data_path()

    def _get_converted_data_path(self) -> str:
        """get name of final saved file to check if it's been already generated"""
        text_to_add = 'converted_crop%i_nChannels%i' % (
            self.crop_size, len(self.used_channels))
        return utils.add_text_to_beginning_of_file_path(self.data_path, text_to_add)

    def get_unsplitted_dataset(self) -> Dataset:
        """get preprocessed dataset, prior to outlier-inlier splitting"""
        # check if preprocessing has already been done
        unsplitted_data_path = utils.add_text_to_beginning_of_file_path(self.converted_data_path, 'unsplitted')
        if os.path.exists(unsplitted_data_path):
            return pd.read_pickle(unsplitted_data_path)
        # useful to avoid frameToInput to perform the transformation of Dataframe to a pickle dict
        data_path = self.data_path
        unprocessed_unsplitted_data_path = utils.add_text_to_beginning_of_file_path(
            self.data_path, 'unprocessed_unsplitted')
        if os.path.exists(unprocessed_unsplitted_data_path):
            data_path = unprocessed_unsplitted_data_path

        # params for Frame input ztf loader and preprocessing
        params = {
            param_keys.DATA_PATH_TRAIN: data_path,
            param_keys.BATCH_SIZE: 0,
            param_keys.CHANNELS_TO_USE: self.used_channels,
            param_keys.TEST_SIZE: 0,  # not used
            param_keys.VAL_SIZE: 0,  # not used
            param_keys.NANS_TO: 0,
            param_keys.CROP_SIZE: self.crop_size,
            param_keys.CONVERTED_DATA_SAVEPATH: unprocessed_unsplitted_data_path
        }
        # instantiate loader, set preprocessor, load dataset
        data_loader = FrameToInput(params)
        data_loader.dataset_preprocessor.set_pipeline(
            [data_loader.dataset_preprocessor.check_single_image,
             data_loader.dataset_preprocessor.clean_misshaped,
             data_loader.dataset_preprocessor.select_channels,
             data_loader.dataset_preprocessor.normalize_by_image,
             data_loader.dataset_preprocessor.nan_to_num,
             data_loader.dataset_preprocessor.crop_at_center
             ])
        dataset = data_loader.get_single_dataset()
        utils.save_pickle(dataset, unsplitted_data_path)
        return dataset

    # TODO: check what happens inside here, particularly correct label consistency when new_labels are defined
    def get_outlier_detection_datasets(self):
        """get outlier trainval test sets, by slecting class 4 as outliers (bogus in ZTF) and generating a train-val
        set of only inliers, while a test set with half-half inliers and outliers"""
        outlier_data_path = utils.add_text_to_beginning_of_file_path(
            self.converted_data_path, 'outlier')
        if os.path.exists(outlier_data_path):
            return pd.read_pickle(outlier_data_path)

        dataset = self.get_unsplitted_dataset()
        # labels from 5 classes to 0-1 as bogus-real
        bogus_class_indx = 4
        new_labels = (dataset.data_label.flatten() != bogus_class_indx) * 1.0
        # print(np.unique(new_labels, return_counts=True))
        inlier_task = 1
        # separate data into train-val-test
        outlier_indexes = np.where(new_labels != inlier_task)[0]
        inlier_indexes = np.where(new_labels == inlier_task)[0]
        # real == inliers
        val_size_inliers = int(np.round(np.sum(
            (new_labels == inlier_task)) * self.val_inlier_percentage))
        np.random.RandomState(seed=self.random_seed).shuffle(inlier_indexes)
        # train-val indexes inlier indexes
        train_inlier_idxs = inlier_indexes[val_size_inliers:]
        val_inlier_idxs = inlier_indexes[:val_size_inliers]
        # train-test inlier indexes
        n_outliers = np.sum(new_labels != inlier_task)
        train_inlier_idxs = train_inlier_idxs[n_outliers:]
        test_inlier_idxs = train_inlier_idxs[:n_outliers]

        X_train, y_train = dataset.data_array[train_inlier_idxs], new_labels[
            train_inlier_idxs]
        X_val, y_val = dataset.data_array[val_inlier_idxs], new_labels[
            val_inlier_idxs]
        X_test, y_test = np.concatenate(
            [dataset.data_array[test_inlier_idxs],
             dataset.data_array[outlier_indexes]]), np.concatenate(
            [new_labels[test_inlier_idxs], new_labels[outlier_indexes]])
        # print('train: ', np.unique(y_train, return_counts=True))
        # print('val: ', np.unique(y_val, return_counts=True))
        # print('test: ', np.unique(y_test, return_counts=True))
        sets_tuple = ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        utils.save_pickle(sets_tuple, outlier_data_path)
        return sets_tuple

    #TODO: implement transformation loading

if __name__ == "__main__":
    """
    parameters = {
        param_keys.DATA_PATH_TRAIN: os.path.join(PROJECT_PATH, '..', 'datasets',
                                                 'ZTF', 'stamp_classifier',
                                                 'ztf_dataset.pkl'),
        param_keys.TEST_SIZE: 100,
        param_keys.VAL_SIZE: 50,
        param_keys.BATCH_SIZE: 50,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.NANS_TO: 1,
        param_keys.INPUT_IMAGE_SIZE: 63,
    }
    data_loader = ZTFLoader(parameters)
    datasets_dict = data_loader.get_datasets()
    print('train %s' % str(datasets_dict[general_keys.TRAIN].data_array.shape))
    print('test %s' % str(datasets_dict[general_keys.TEST].data_array.shape))
    print('val %s' % str(datasets_dict[general_keys.VALIDATION].data_array.shape))

    # replication test
    train_dataset = datasets_dict[general_keys.TRAIN]
    print('train samples %s, train SNe %i' % (str(train_dataset.data_array.shape),
                                              int(np.sum(
                                                  train_dataset.data_label))))
    train_dataset.balance_data_by_replication_2_classes()
    print('train samples %s, train SNe %i' % (str(train_dataset.data_array.shape),
                                              int(np.sum(
                                                  train_dataset.data_label))))
    """
    print('')
