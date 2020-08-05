"""
Test if table of pipeline (best model) hold for refactored versions
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from models.streaming_geotransform.geotransform_base_dirichlet_alphas_save \
    import GeoTransformBaseDirichletAlphasSaved
from modules import utils
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.data_loaders.ztf_small_outlier_loader import \
    ZTFSmallOutlierLoader
from parameters import loader_keys, general_keys
from modules.geometric_transform. \
    streaming_transformers.transformer_ranking import RankingTransformer
from modules.print_manager import PrintManager
from tests.test_streaming_alphas_refact_classic_transform_od import \
    print_mean_results, get_best_transformation_tuples
from models.streaming_geotransform.geotransform_alphas_wrn_wait_first_epoch \
    import GeoTransformAlphasWRN1Epoch
from typing import Callable
from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')

def fit_and_evaluate_model_n_times_alphas(
    ModelClass: Callable[[], GeoTransformAlphasWRN1Epoch],
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
            wait_first_epoch=True, verbose=verbose)
        results_i = model.evaluate(
            x_test, y_test, data_loader_name, 'real',
            save_metrics=True, save_histogram=True, get_auroc_acc_only=True,
            verbose=verbose)
        result_dicts.append(results_i)
        del model
    print('\nResults %i trains, Model: %s, Transformer: %s, Data: %s' % (
        n_times, model.name, transformer.name, data_loader_name
    ))
    print_mean_results(result_dicts)
    return result_dicts

if __name__ == '__main__':
    EPOCHS = 1000
    ITERATIONS_TO_VALIDATE = 100
    PATIENCE = 10
    VERBOSE = False
    TRAIN_N_TIME = 10
    RESULTS_FOLDER_NAME = 'large_datasets_best_transforms_small_validate_' \
                          'step_wait_first_epoch_no_saving'

    utils.set_soft_gpu_memory_growth()

    ztf_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH,
            '../datasets/ALeRCE_data/all_ztf_dataset_tuples.pkl'),
    }
    ztf_loader = ZTFSmallOutlierLoader(
        ztf_params, 'large_ztf', pickles_usage=False)
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 100000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    hits_loader = HiTSOutlierLoader(
        hits_params, 'large_hits', pickles_usage=False)
    data_loaders = [
        ztf_loader,
        hits_loader
    ]
    transformer = RankingTransformer()
    test_folder_path = os.path.join(
        PROJECT_PATH, 'results', RESULTS_FOLDER_NAME)
    utils.check_path(test_folder_path)
    for outlier_loader in data_loaders:
        transformer.set_transformations_to_perform(
            get_best_transformation_tuples(outlier_loader))
        print_manager = PrintManager()
        test_log_path = os.path.join(
            test_folder_path, '%s.log' % outlier_loader.name)
        log_file = open(test_log_path, 'w')
        print_manager.file_printing(log_file)
        print('N_transforms: %i' % (transformer.n_transforms))
        (x_train, y_train), (x_val, y_val), (
            x_test, y_test) = outlier_loader.get_outlier_detection_datasets()
        print(np.unique(y_test, return_counts=True))
        parameters = (EPOCHS, ITERATIONS_TO_VALIDATE, PATIENCE, VERBOSE,
                      RESULTS_FOLDER_NAME, outlier_loader.name)
        data_tuples = ((x_train, y_train), (x_val, y_val), (
            x_test, y_test))
        fit_and_evaluate_model_n_times_alphas(
            GeoTransformAlphasWRN1Epoch, transformer, data_tuples, parameters,
            TRAIN_N_TIME)
        print_manager.close()
