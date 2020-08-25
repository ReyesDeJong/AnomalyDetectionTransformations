"""
First attempt (train_step_tf2 like) to build geometric trasnformer for outlier detection in tf2
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
from models.streaming_geotransform.geotransform_base_dirichlet_alphas_save \
    import GeoTransformBaseDirichletAlphasSaved
from modules.networks.streaming_network. \
    streaming_wide_resnet_wait_first_epoch import StreamingWideResnetWait1Epoch
from modules.print_manager import PrintManager


class GeoTransformAlphasWRN1Epoch(GeoTransformBaseDirichletAlphasSaved):

    def __init__(self, n_channels: int, transformer: AbstractTransformer,
        results_folder_name=None, name='GeoTransform_Alphas_WRN_Wait_1_Epoch'):
        classifier = StreamingWideResnetWait1Epoch(
            n_channels, transformer)
        super().__init__(classifier, transformer, results_folder_name, name)

    def fit(self, x_train, epochs, x_validation=None, batch_size=128,
        iterations_to_validate=None, patience=None, wait_first_epoch=False,
        verbose=True, iterations_to_print_train=None):
        print_manager = PrintManager().verbose_printing(verbose)
        print(self.name)
        if epochs is None:
            epochs = self._get_original_paper_epochs()
            patience = int(1e100)
        self.classifier.fit(
            x_train, epochs, x_validation, batch_size,
            iterations_to_validate,
            patience, verbose, wait_first_epoch,
            iterations_to_print_train=iterations_to_print_train)
        self._update_dirichlet_alphas(x_train, verbose)
        if x_validation is None:
            x_validation = x_train
        self._update_prediction_threshold(x_validation, verbose)
        self.save_model(self.classifier.best_model_weights_path)
        print_manager.close()


if __name__ == '__main__':
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from modules.data_loaders.ztf_small_outlier_loader import \
        ZTFSmallOutlierLoader
    from parameters import loader_keys, general_keys
    from modules.geometric_transform. \
        streaming_transformers.transformer_original72 import Original72Transformer
    from modules import utils
    import matplotlib

    matplotlib.use('Agg')

    EPOCHS = 1000
    ITERATIONS_TO_VALIDATE = 10
    ITERATIONS_TO_PRINT_TRAIN = 10
    PATIENCE = 0
    VERBOSE = True

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
    outlier_loader = ztf_loader
    # outlier_loader = hits_loader

    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = outlier_loader.get_outlier_detection_datasets()
    transformer = Original72Transformer()
    model = GeoTransformAlphasWRN1Epoch(
        n_channels=x_train.shape[-1],
        transformer=transformer,
        results_folder_name='test_base')
    model.fit(
        x_train,
        #epochs=EPOCHS,
        #x_validation=x_val,
        iterations_to_validate=ITERATIONS_TO_VALIDATE, patience=PATIENCE,
        verbose=VERBOSE, iterations_to_print_train=ITERATIONS_TO_PRINT_TRAIN)
    model.evaluate(
        x_test, y_test, outlier_loader.name, 'real', save_metrics=True,
        save_histogram=True, get_specific_metrics=True, verbose=VERBOSE)
