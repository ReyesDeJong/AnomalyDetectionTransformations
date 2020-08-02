"""
First attempt (train_step_tf2 like) to build geometric trasnformer for outlier detection in tf2
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

import tensorflow as tf
from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
from parameters import general_keys
import numpy as np
from modules import dirichlet_utils, utils
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from modules.metrics import accuracies_by_threshold, accuracy_at_thr
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
# from joblib import Parallel, delayed
import time
from modules.print_manager import PrintManager
from modules.networks.streaming_network.streaming_transformations_deep_hits\
    import StreamingTransformationsDeepHits
from models.streaming_geotransform.geotransform_base import GeoTransformBase
import matplotlib
matplotlib.use('Agg')

# this model must have streaming clf as input
class GeoTransformBaseNotAllTransformsAtOnce(GeoTransformBase):
    def __init__(self, classifier: StreamingTransformationsDeepHits,
        transformer: AbstractTransformer, results_folder_name=None,
        name='GeoTransform_Base_Not_All_Trfs'):
        super().__init__(classifier, transformer, results_folder_name, name)

    def _predict_matrix_probabilities(self, x_data, transform_batch_size=512,
        predict_batch_size=1024):
        return

    # def _get_prediction_for_specific_transform_index(self,
    #     x_train, x_eval, transformation_index,
    #     transform_batch_size, predict_batch_size):
    #     x_train_transformed, _ = self.transformer.apply_transforms(
    #         x_train, [transformation_index], transform_batch_size)
    #     predictions_train = self.classifier.predict(
    #         x_train_transformed, predict_batch_size)
    #     x_eval_transformed, _ = self.transformer.apply_transforms(
    #         x_eval, [transformation_index], transform_batch_size)
    #     predictions_eval = self.classifier.predict(
    #         x_eval_transformed, predict_batch_size)
    #     return predictions_train, predictions_eval

    def _get_prediction_for_specific_transform_index(self,
        x_train, x_eval, transformation_index,
        transform_batch_size, predict_batch_size):
        x_concat = np.concatenate([x_train, x_eval], axis=0)
        x_concat_transformed, _ = self.transformer.apply_transforms(
            x_concat, [transformation_index], transform_batch_size)
        predictions_concat = self.classifier.predict(
            x_concat_transformed, predict_batch_size)
        predictions_train = predictions_concat[:len(x_train)]
        predictions_eval = predictions_concat[len(x_train):]
        return predictions_train, predictions_eval

    # TODO: save dirichlet params
    def predict_dirichlet_score(self, x_train, x_eval,
        transform_batch_size=512, predict_batch_size=1024, verbose=True):
        print_manager = PrintManager().verbose_printing(verbose)
        n_transforms = self.transformer.n_transforms
        # get actual length if all transformations applied on model input
        dirichlet_scores = np.zeros(len(x_eval))
        print('Calculating dirichlet scores...')
        for t_ind in tqdm(range(n_transforms), disable=not verbose):
            observed_dirichlet, x_eval_p = self.\
                _get_prediction_for_specific_transform_index(
                x_train, x_eval, t_ind, transform_batch_size,
                predict_batch_size)
            dirichlet_scores += dirichlet_utils.dirichlet_score(
                observed_dirichlet, x_eval_p)
            assert np.isfinite(dirichlet_scores).all()
        dirichlet_scores /= n_transforms
        print_manager.close()
        return dirichlet_scores

if __name__ == '__main__':
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from modules.data_loaders.ztf_small_outlier_loader import \
        ZTFSmallOutlierLoader
    from parameters import loader_keys
    from modules.geometric_transform.\
        streaming_transformers.transformer_ranking import RankingTransformer
    EPOCHS = 1000
    ITERATIONS_TO_VALIDATE = 0
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

    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = outlier_loader.get_outlier_detection_datasets()
    transformer = RankingTransformer()
    # transformer.set_transformations_to_perform(
    #     transformer.transformation_tuples*5)
    clf = StreamingTransformationsDeepHits(transformer)
    model = GeoTransformBaseNotAllTransformsAtOnce(
        classifier=clf, transformer=transformer,
        results_folder_name='test_base')
    model.fit(
        x_train, epochs=EPOCHS, x_validation=x_val,
        iterations_to_validate=ITERATIONS_TO_VALIDATE, patience=PATIENCE,
        verbose=VERBOSE)
    model.evaluate(
        x_train, x_test, y_test, outlier_loader.name, 'real', x_val,
        save_metrics=False, save_histogram=False, get_auroc_acc_only=True,
        verbose=VERBOSE)
    model.evaluate(
        x_train, x_test, y_test, outlier_loader.name, 'real', x_val,
        save_metrics=False, save_histogram=False, get_auroc_acc_only=True,
        verbose=VERBOSE)
