"""
Test if table of pipeline (best model) hold for refactored versions
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from models.streaming_geotransform.geotransform_alphas_wrn import \
    GeoTransformAlphasWRN
from modules import utils
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.data_loaders.ztf_small_outlier_loader import \
    ZTFSmallOutlierLoader
import numpy as np
from parameters import loader_keys, general_keys
from modules.geometric_transform. \
    streaming_transformers.transformer_ranking import RankingTransformer
from tqdm import tqdm
from typing import Callable, List
from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer


# from modules.networks.streaming_network. \
#     streaming_transformations_wide_resnet import \
#     StreamingTransformationsWideResnet

def print_mean_results(result_dicts: List[dict]):
    dict_keys = result_dicts[0].keys()
    all_results_in_lists_dict = {}
    for key_i in dict_keys:
        all_results_in_lists_dict[key_i] = []
        for dict_i in result_dicts:
            all_results_in_lists_dict[key_i].append(dict_i[key_i])
    message = ''
    for key_i in all_results_in_lists_dict:
        message += '%s %.6f +/- %.6f, ' % (
            key_i, np.mean(all_results_in_lists_dict[key_i]),
            np.std(all_results_in_lists_dict[key_i]))
    message = message[:-2]
    print(message)


def fit_and_evaluate_model_n_times(
    ModelClass: Callable[[], GeoTransformAlphasWRN],
    transformer: AbstractTransformer, data_tuples, parameters, n_times):
    epochs, iterations_to_validate, patience, verbose, results_folder_name, \
    data_loader_name = parameters
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = data_tuples
    result_dicts = []
    for _ in tqdm(range(n_times)):
        model = ModelClass(
            n_channels=x_train.shape[:-1], transformer=transformer,
            results_folder_name=results_folder_name)
        model.fit(
            x_train, epochs=epochs, x_validation=x_val,
            iterations_to_validate=iterations_to_validate, patience=patience,
            verbose=verbose)
        results_i = model.evaluate(
            x_test, y_test, data_loader_name, 'real', x_val,
            save_metrics=True, save_histogram=False, get_auroc_acc_only=True,
            verbose=verbose)
        result_dicts.append(results_i)
    print('Results %i trains, Model: %s, Transformer: %s, Data: %s' % (
        n_times, model.name, transformer.name, data_loader_name
    ))
    print_mean_results(result_dicts)
    return result_dicts


if __name__ == '__main__':
    EPOCHS = 1
    # 000
    ITERATIONS_TO_VALIDATE = 0
    PATIENCE = 0
    VERBOSE = False
    TRAIN_N_TIME = 3

    utils.set_soft_gpu_memory_growth()

    ztf_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH,
            '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
    }
    ztf_loader = ZTFSmallOutlierLoader(ztf_params)
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 10000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    hits_loader = HiTSOutlierLoader(hits_params)
    # outlier_loader = ztf_loader
    outlier_loader = hits_loader

    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = outlier_loader.get_outlier_detection_datasets()
    transformer = RankingTransformer()
    transformer.set_transformations_to_perform(
        transformer.transformation_tuples[:3])

    parameters = (EPOCHS, ITERATIONS_TO_VALIDATE, PATIENCE, VERBOSE,
                  'test_n_time_base', outlier_loader.name)
    data_tuples = ((x_train, y_train), (x_val, y_val), (
        x_test, y_test))
    fit_and_evaluate_model_n_times(
        GeoTransformAlphasWRN, transformer, data_tuples, parameters,
        TRAIN_N_TIME)
