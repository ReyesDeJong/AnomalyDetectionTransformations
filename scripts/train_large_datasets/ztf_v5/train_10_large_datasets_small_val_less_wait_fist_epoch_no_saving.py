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
    streaming_transformers.transformer_ranking import RankingTransformer
from modules.print_manager import PrintManager
from tests.test_streaming_alphas_refact_classic_transform_od import \
    get_best_transformation_tuples
from models.streaming_geotransform.geotransform_alphas_wrn_wait_first_epoch \
    import GeoTransformAlphasWRN1Epoch
from scripts.train_large_datasets. \
    train_10_large_datasets_small_val_less_wait_fist_epoch_no_saving import \
    fit_and_evaluate_model_n_times_alphas
import numpy as np
import matplotlib

matplotlib.use('Agg')

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
            '../datasets/ALeRCE_data/v5_big_ztf_dataset_tuples.pkl'),
    }
    ztf_loader = ZTFSmallOutlierLoader(
        ztf_params, 'v5_ztf', pickles_usage=False)
    data_loaders = [
        ztf_loader,
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
