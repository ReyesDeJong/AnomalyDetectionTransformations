# %%

"""
Stamp classifier datasets are generated, which are saved to process by geotransform.
Dsitinguishing between Alerce-Ashish bogus.

Geotransform datasets are generated in geotransform repo, here data is just pickled
because geotransform repo would suffer to much variations to be able to perform this.
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from modules.data_loaders.frame_to_input_with_features import FrameToInputWithFeatures
from parameters import param_keys, general_keys
import numpy as np
import pandas as pd
from modules import utils
from modules.data_set_generic import Dataset
from typing import List
from data_generator.identification_of_ashish_bogus_in_training_df \
    import get_df_with_data_source_indicated

def get_preprocessed_dataset(params, alerce_df_path, ashish_bogus_df_path,
    save_folder=''):
    # including data source (alerce-ashish) to df
    df_with_clear_source = get_df_with_data_source_indicated(
        alerce_df_path, ashish_bogus_df_path)

    # including indexes to df
    dataframe = df_with_clear_source
    dataframe['indexes'] = dataframe.index.astype(int)

    #loading_data
    data_loader = FrameToInputWithFeatures(params)
    data_loader.set_dumping_data_to_pickle(dump_to_pickle=False)
    data_loader.df = dataframe
    data_loader.dataset_preprocessor.set_pipeline(
        [
            data_loader.dataset_preprocessor.image_check_single_image,
            data_loader.dataset_preprocessor.image_clean_misshaped,
            data_loader.dataset_preprocessor.image_select_channels,
            data_loader.dataset_preprocessor.image_crop_at_center,
            data_loader.dataset_preprocessor.image_normalize_by_image_1_1,
            data_loader.dataset_preprocessor.image_nan_to_num,
        ]
    )
    datasets_dict = data_loader.get_preprocessed_datasets_splitted()
    train_set = datasets_dict[general_keys.TRAIN]
    val_set = datasets_dict[general_keys.VALIDATION]
    test_set = datasets_dict[general_keys.TEST]
    return train_set, val_set, test_set

if __name__ == "__main__":
    SAVE_DATA = True
    RANDOM_SEED = 40
    data_name = 'training_set_Aug-07-2020.pkl'
    data_folder = "/home/ereyes/Projects/Thesis/stamp_classifier_updater/data/"
    alerce_df_path = os.path.join(data_folder, data_name)
    ashish_bogus_df_path = os.path.join(data_folder, 'bogus_ashish.pkl')
    save_folder = '/home/ereyes/Projects/Thesis/datasets/thesis_data/ztfv7_stamp_clf_data'
    utils.check_path(save_folder)

    # Data loader params
    n_classes = 5
    params = {
        param_keys.BATCH_SIZE: None,
        param_keys.UNDERSAMPLING: 20000,
        param_keys.CONVERTED_DATA_SAVEPATH: None,
        param_keys.DATA_PATH_TRAIN: None,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TEST_SIZE: n_classes * 200,
        param_keys.VAL_SIZE: n_classes * 100,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: 21,
        param_keys.INPUT_IMAGE_SIZE: 21,
        param_keys.VALIDATION_RANDOM_SEED: RANDOM_SEED,
        param_keys.TEST_RANDOM_SEED: RANDOM_SEED,
        param_keys.FEATURES_NAMES_LIST: [
            'indexes', 'data_source', 'oid', 'sgscore1', 'distpsnr1',
            'sgscore2', 'distpsnr2', 'sgscore3', 'distpsnr3',
            'isdiffpos', 'fwhm', 'magpsf', 'sigmapsf', 'ra', 'dec',
            'diffmaglim',
            'rb', 'distnr', 'magnr', 'classtar', 'ndethist', 'ncovhist',
            'ecl_lat',
            'ecl_long', 'gal_lat', 'gal_long', 'non_detections', "chinr",
            "sharpnr"],
        param_keys.FEATURES_CLIPPING_DICT: {
            "sgscore1": [-1, "max"],
            "distpsnr1": [-1, "max"],
            "sgscore2": [-1, "max"],
            "distpsnr2": [-1, "max"],
            "sgscore3": [-1, "max"],
            "distpsnr3": [-1, "max"],
            "fwhm": ["min", 10],
            "distnr": [-1, "max"],
            "magnr": [-1, "max"],
            "ndethist": ["min", 20],
            "ncovhist": ["min", 3000],
            "chinr": [-1, 15],
            "sharpnr": [-1, 1.5],
            "non_detections": ["min", 2000]
        }
    }

    train_set, val_set, test_set = get_preprocessed_dataset(
        params, alerce_df_path, ashish_bogus_df_path)

    print('label_counts')
    print('Train %s' % str(np.unique(train_set.data_label, return_counts=True)))
    print('Val %s' % str(np.unique(val_set.data_label, return_counts=True)))
    print('Test %s' % str(np.unique(test_set.data_label, return_counts=True)))
    print('source_counts')
    print('Train %s' % str(np.unique(train_set.meta_data[:,1], return_counts=True)))
    print('Val %s' % str(np.unique(val_set.meta_data[:,1], return_counts=True)))
    print('Test %s' % str(np.unique(test_set.meta_data[:,1], return_counts=True)))
    print(train_set.meta_data[0])

    # data in sets and dicts
    dataset_dict = {
        'Train': {'images': train_set.data_array,
                  'labels': train_set.data_label,
                  'features': train_set.meta_data},
        'Validation': {'images': val_set.data_array,
                       'labels': val_set.data_label,
                       'features': val_set.meta_data},
        'Test': {'images': test_set.data_array, 'labels': test_set.data_label,
                 'features': test_set.meta_data}
    }

    # saving data
    if SAVE_DATA:
        # stamps clf data
        save_path = os.path.join(save_folder,
                                 'ztfv7_stamp_clf_processed_undersampled.pkl')
        utils.save_pickle(dataset_dict, save_path)
