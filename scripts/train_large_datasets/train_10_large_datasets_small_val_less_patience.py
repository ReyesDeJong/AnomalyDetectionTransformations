"""
Test if table of pipeline (best model) hold for refactored versions
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from models.streaming_geotransform.geotransform_alphas_wrn import \
    GeoTransformAlphasWRN
from modules import utils
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.data_loaders.ztf_small_outlier_loader import \
    ZTFSmallOutlierLoader
from parameters import loader_keys, general_keys
from modules.geometric_transform. \
    streaming_transformers.transformer_ranking import RankingTransformer
from modules.print_manager import PrintManager
from tests.test_streaming_alphas_refact_classic_transform_od import \
    get_best_transformation_tuples, fit_and_evaluate_model_n_times_alphas
import numpy as np
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    EPOCHS = 1000
    ITERATIONS_TO_VALIDATE = 500
    PATIENCE = 20
    VERBOSE = False
    TRAIN_N_TIME = 10
    RESULTS_FOLDER_NAME = 'large_datasets_best_transforms_small_validate_step'

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
            GeoTransformAlphasWRN, transformer, data_tuples, parameters,
            TRAIN_N_TIME)
        print_manager.close()
