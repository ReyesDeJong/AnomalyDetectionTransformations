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
from parameters import general_keys
import numpy as np
from modules import dirichlet_utils, utils
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from modules.metrics import accuracies_by_threshold, accuracy_at_thr
from tqdm import tqdm
from models.streaming_geotransform.geotransform_base import GeoTransformBase
import matplotlib.pyplot as plt
# from joblib import Parallel, delayed
import time
from modules.networks.streaming_network.streaming_transformations_deep_hits \
    import StreamingTransformationsDeepHits
import matplotlib

matplotlib.use('Agg')


# this model must have streaming clf as input
class GeoTransformBaseDirichletAlphasSaved(GeoTransformBase):
    def __init__(self, classifier: StreamingTransformationsDeepHits,
        transformer: AbstractTransformer, results_folder_name=None,
        name='GeoTransform_Base_Dirichlet_Alphas_Saved'):
        super().__init__(classifier, transformer, results_folder_name, name)
        self.dirichlet_alphas = self._initialize_dirichlet_alphas()

    def _initialize_dirichlet_alphas(self):
        n_transforms = self.transformer.n_transforms
        return np.random.normal(
            size=(n_transforms, n_transforms)) * 15000

    def _update_dirichlet_alphas(self, x_data, verbose):
        transform_batch_size = 512
        predict_batch_size = 1024
        n_transforms = self.transformer.n_transforms
        dirichlet_alphas = np.zeros((n_transforms, n_transforms))
        print('Calculating dirichlet alphas...')
        for t_ind in tqdm(range(n_transforms), disable=not verbose):
            x_data_transformed, _ = self.transformer.apply_transforms(
                x_data, [t_ind], transform_batch_size)
            predictions_data = self.classifier.predict(
                x_data_transformed, predict_batch_size)
            dirichlet_alphas[t_ind, :] = dirichlet_utils.get_mle_alpha_t(
                predictions_data)
        self.dirichlet_alphas = dirichlet_alphas

    def fit(self, x_train, epochs, x_validation=None, batch_size=128,
        iterations_to_validate=None, patience=None, verbose=True):
        if epochs is None:
            epochs = self._get_original_paper_epochs()
            patience = int(1e100)
        self.classifier.fit(
            x_train, epochs, x_validation, batch_size, iterations_to_validate,
            patience, verbose)
        self._update_dirichlet_alphas(x_train, verbose)

    # TODO: avoid apply_all_transforms at once
    def _predict_matrix_probabilities(self, x_data, transform_batch_size=512,
        predict_batch_size=1024):
        return

    def predict_dirichlet_score(self, x_eval, transform_batch_size=512,
        predict_batch_size=1024, verbose=True):
        self.print_manager.verbose_printing(verbose)
        n_transforms = self.transformer.n_transforms
        dirichlet_scores = np.zeros(len(x_eval))
        print('Calculating dirichlet scores...')
        for t_ind in tqdm(range(n_transforms), disable=not verbose):
            x_data_transformed, _ = self.transformer.apply_transforms(
                x_eval, [t_ind], transform_batch_size)
            predictions_data = self.classifier.predict(
                x_data_transformed, predict_batch_size)
            mle_alpha_t = self.dirichlet_alphas[t_ind, :]
            x_eval_p = dirichlet_utils.correct_0_value_predictions(
                predictions_data)
            dirichlet_scores += dirichlet_utils.dirichlet_normality_score(
                mle_alpha_t, x_eval_p)
            assert np.isfinite(dirichlet_scores).all()
        dirichlet_scores /= n_transforms
        self.print_manager.close()
        return dirichlet_scores

    def evaluate(self, x_eval, y_eval, dataset_name,
        class_name='inlier', x_validation=None, transform_batch_size=512,
        evaluation_batch_size=1024, save_metrics=False,
        additional_score_save_path_list=None, save_histogram=False,
        get_auroc_acc_only=False, verbose=True):
        self.print_manager.verbose_printing(verbose)
        print('\nEvaluating model...')
        start_time = time.time()
        # validatioon is ussed to set accuracy thresholds
        if x_validation is None:
            x_validation = x_eval
        dirichlet_scores_eval = self.predict_dirichlet_score(
            x_eval, transform_batch_size, evaluation_batch_size, verbose)
        dirichlet_scores_validation = self.predict_dirichlet_score(
            x_validation, transform_batch_size, evaluation_batch_size, verbose)
        metrics = self._get_metrics_dict(
            dirichlet_scores_eval, dirichlet_scores_validation, y_eval)
        self._save_histogram(
            metrics, general_keys.DIRICHLET, dataset_name, class_name,
            save_histogram)
        self._print_final_metrics(
            metrics, keys_to_keep=[
                'roc_auc', 'acc_at_percentil', 'pr_auc_norm'])
        metrics = self._filter_metrics_to_return(
            metrics, get_auroc_acc_only,
            keys_to_keep=['roc_auc', 'acc_at_percentil'])
        metrics_save_path = self.get_metrics_save_path(
            general_keys.DIRICHLET, dataset_name, class_name)
        self._save_metrics_on_paths(
            additional_score_save_path_list, metrics_save_path,
            metrics, save_metrics)
        print("\nEvaluation time: %s" % utils.timer(
                start_time, time.time()))
        self.print_manager.close()
        return metrics

    def save_weights(self, path):
        self.classifier.save_weights(path)

    def load_weights(self, path, by_name=False):
        self.classifier.load_weights(path, by_name=by_name)


if __name__ == '__main__':
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from modules.data_loaders.ztf_small_outlier_loader import \
        ZTFSmallOutlierLoader
    from parameters import loader_keys
    from modules.geometric_transform. \
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
    clf = StreamingTransformationsDeepHits(transformer)
    model = GeoTransformBaseDirichletAlphasSaved(
        classifier=clf, transformer=transformer,
        results_folder_name='test_base')
    model.fit(
        x_train, epochs=EPOCHS, x_validation=x_val,
        iterations_to_validate=ITERATIONS_TO_VALIDATE, patience=PATIENCE,
        verbose=VERBOSE)
    model.evaluate(
        x_test, y_test, outlier_loader.name, 'real', x_val, save_metrics=True,
        save_histogram=True, get_auroc_acc_only=True, verbose=VERBOSE)
    del model
    # model = GeoTransformBase(
    #     classifier=clf, transformer=transformer,
    #     results_folder_name=None)
    # model_weights_folder = os.path.join(
    #     PROJECT_PATH,
    #     'results/test_base/GeoTransform_Base_DH_'
    #     'Streaming_Trfs_20200729-230251/')
    # model_weights_path = os.path.join(model_weights_folder, 'checkpoints',
    #                                   'best_weights.ckpt')
    # model.load_weights(model_weights_path)
    # model.set_model_results_path(model_weights_folder)
    # model.evaluate(
    #     x_test, y_test, outlier_loader.name, 'real', x_val,
    #     save_metrics=False, save_histogram=True, get_auroc_acc_only=True,
    #     verbose=VERBOSE)
