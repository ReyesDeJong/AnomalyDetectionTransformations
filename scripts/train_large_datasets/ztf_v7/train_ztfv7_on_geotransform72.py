"""
Test if table of pipeline (best model) hold for refactored versions
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

from modules import utils
from modules.data_loaders.ztf_small_outlier_loader import \
    ZTFSmallOutlierLoader
from parameters import loader_keys
from modules.geometric_transform. \
    streaming_transformers.transformer_original72 import Original72Transformer
from modules.print_manager import PrintManager
from models.streaming_geotransform.geotransform_alphas_wrn_wait_first_epoch \
    import GeoTransformAlphasWRN1Epoch
from tests.test_streaming_alphas_refact_classic_transform_od import \
    print_mean_results
from typing import Callable, List
from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
import matplotlib
from tqdm import trange
from scripts.train_large_datasets.ztf_v7. \
    train_ztfv7_different_sampling_strategies import get_test_data_loaders

matplotlib.use('Agg')


def fit_and_evaluate_model_n_times_alphas(
    ModelClass: Callable[[], GeoTransformAlphasWRN1Epoch],
    transformer: AbstractTransformer, model_params: dict,
    train_loader: ZTFSmallOutlierLoader,
    test_loaders: List[ZTFSmallOutlierLoader], n_times, results_folder_name,
    model_base_name):
    (x_train, y_train), (x_val, y_val), _ = train_loader. \
        get_outlier_detection_datasets()
    result_dicts = {test_loader_i.name: [] for test_loader_i in test_loaders}
    for _ in trange(n_times):
        model = ModelClass(
            n_channels=x_train.shape[:-1], transformer=transformer,
            results_folder_name=results_folder_name,
            name='%s_%s' % (train_loader.name, model_base_name))
        if model_params['x_validation'] is None:
            model.fit(
                x_train=x_train,
                # x_validation=x_val,
                **model_params)
        else:
            model.fit(
                x_train=x_train,
                x_validation=x_val,
                **model_params)
        for test_loader_i in test_loaders:
            _, _, (x_test, y_test) = test_loader_i. \
                get_outlier_detection_datasets()
            results_i = model.evaluate(
                x_test, y_test, test_loader_i.name, 'real',
                save_metrics=True, save_histogram=True,
                get_specific_metrics=True, verbose=model_params['verbose'])
            result_dicts[test_loader_i.name].append(results_i)
    for test_loader_i in test_loaders:
        print('\nResults %i trains, Model: %s, Transformer: %s, Data: %s' % (
            n_times, model.name, transformer.name, test_loader_i.name
        ))
        print_mean_results(result_dicts[test_loader_i.name])
    return result_dicts


if __name__ == '__main__':
    # Training dataset sampling strategies names
    train_dataset_file_names = [
        # 'new_small_od_dataset_tuples.pkl',
        'v7_ztf_disjoint_test.pkl',
        # 'v7_ztf_oversample_disjoint_test.pkl',
        # 'v7_ztf_undersample_disjoint_test.pkl',
        # 'v7_ztf_under_over_sample_disjoint_test.pkl'
    ]

    # Base parameters
    EPOCHS = 1000
    VERBOSE = False
    TRAIN_N_TIME = 10
    RESULTS_FOLDER_NAME = 'ztf_v7_geotransform72'

    # Set soft growth of gpus
    utils.set_soft_gpu_memory_growth()

    # Get test data_loaders
    datesets_alerce_folder_path = os.path.join(
        PROJECT_PATH, '../datasets/ALeRCE_data')
    test_set_paths = [
        'ztf_v7/v7_ztf_disjoint_test.pkl', 'new_small_od_dataset_tuples.pkl']
    test_loaders = get_test_data_loaders(
        datesets_alerce_folder_path, test_set_paths)

    # Model Parameters Dict {'Base_name': {params}}
    model_params_dict = {
        'GeoTrfSOTA': {
            'epochs': None,
            'x_validation': None,
            'iterations_to_validate': None,
            'patience': None,
            'wait_first_epoch': None,
            'verbose': VERBOSE,
            'iterations_to_print_train': 50},
        'GeoTrfWaitFirstEpochValEvery50': {
            'epochs': EPOCHS,
            'iterations_to_validate': 50,
            'patience': 10,
            'wait_first_epoch': True,
            'verbose': VERBOSE,
            'iterations_to_print_train': 50},
        'GeoTrfWaitFirstEpochValEvery200Patience20': {
            'epochs': EPOCHS,
            'iterations_to_validate': 200,
            'patience': 20,
            'wait_first_epoch': True,
            'verbose': VERBOSE,
            'iterations_to_print_train': 50},
        'GeoTrfEndEpochVal': {
            'epochs': EPOCHS,
            'iterations_to_validate': 0,
            'patience': 0,
            'wait_first_epoch': False,
            'verbose': VERBOSE,
            'iterations_to_print_train': 100},
    }

    # Train all models
    test_folder_path = os.path.join(
        PROJECT_PATH, 'results', RESULTS_FOLDER_NAME)
    utils.check_path(test_folder_path)
    for model_base_name in model_params_dict.keys():
        for train_file_name in train_dataset_file_names:
            # Create ztf train loader
            ztf_params = {
                loader_keys.DATA_PATH: os.path.join(
                    datesets_alerce_folder_path, 'ztf_v7', train_file_name)}
            train_loader = ZTFSmallOutlierLoader(
                ztf_params, train_file_name.split('.')[0], pickles_usage=False)
            (x_train, y_train), (x_val, y_val), _ = \
                train_loader.get_outlier_detection_datasets()
            # print(x_train.shape)
            # print(x_val.shape)
            # print(train_loader.name)
            # Create transformer
            transformer = Original72Transformer()
            # Train model N times
            print_manager = PrintManager()
            test_log_path = os.path.join(
                test_folder_path, '%s_%s.log' % (
                    train_loader.name, model_base_name))
            log_file = open(test_log_path, 'w')
            print_manager.file_printing(log_file)
            print('\nModel %s, Train dataset %s, N_transforms: %i' % (
                model_base_name, train_loader.name, transformer.n_transforms))
            fit_and_evaluate_model_n_times_alphas(
                GeoTransformAlphasWRN1Epoch, transformer,
                model_params_dict[model_base_name], train_loader, test_loaders,
                TRAIN_N_TIME, RESULTS_FOLDER_NAME, model_base_name)
            print_manager.close()
            log_file.close()
