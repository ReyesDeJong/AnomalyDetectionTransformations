"""
Test if table of pipeline (best model) hold for refactored versions
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
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
import pandas as pd

matplotlib.use('Agg')

if __name__ == '__main__':
    CHUNK_I = 1
    EPOCHS = 1000
    ITERATIONS_TO_VALIDATE = 100
    PATIENCE = 10
    VERBOSE = True
    TRAIN_N_TIME = 10
    PREDICTIONS_FOLDER_NAME = os.path.join(
        PROJECT_PATH, 'results/unlabeled_predictions')
    WEIGHTS_FOLDER_NAME = 'large_datasets_best_transforms_small_validate_step' \
                          '_wait_first_epoch'
    utils.set_soft_gpu_memory_growth()

    ztf_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH,
            '../datasets/ALeRCE_data/all_ztf_dataset_tuples.pkl'),
    }
    ztf_loader = ZTFSmallOutlierLoader(
        ztf_params, 'large_ztf', pickles_usage=False)

    unlabaled_datapath_folder = os.path.join(
        PROJECT_PATH, '..', 'datasets/ALeRCE_data')
    unlabaled_file = os.path.join(unlabaled_datapath_folder,
                                  'unlabeled_ztf_chunk_%i.pkl')
    unlabeled_stamps = pd.read_pickle(unlabaled_file)

    weights_path = os.path.join(
        PROJECT_PATH, 'results', WEIGHTS_FOLDER_NAME,
        'GeoTransform_Alphas_WRN_Wait_1_Epoch_'
        'WRN_Streaming_Trfs_20200804-222433', 'checkpoints',
        'best_weights.ckpt'
    )
    transformer = RankingTransformer()
    transformer.set_transformations_to_perform(
        get_best_transformation_tuples(ztf_loader))
    print('N_transforms: %i' % (transformer.n_transforms))
    model = GeoTransformAlphasWRN1Epoch(unlabeled_stamps.shape[-1], transformer)
    model.load_model(weights_path)
    predictions = model.predict(unlabeled_stamps)
    print(np.unique(predictions, return_counts=True))

    utils.check_path(PREDICTIONS_FOLDER_NAME)
    save_prediction_path = os.path.join(
        PREDICTIONS_FOLDER_NAME, 'predictions_chunk%i.pkl' % CHUNK_I)
    utils.save_pickle(predictions, save_prediction_path)
