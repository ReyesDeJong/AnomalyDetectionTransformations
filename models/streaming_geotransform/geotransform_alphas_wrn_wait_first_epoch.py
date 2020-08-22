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