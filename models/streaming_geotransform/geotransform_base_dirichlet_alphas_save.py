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
from tqdm import tqdm
from models.streaming_geotransform.geotransform_base import GeoTransformBase
import time
from modules.networks.streaming_network.streaming_transformations_deep_hits \
    import StreamingTransformationsDeepHits
from modules.print_manager import PrintManager
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from modules.metrics import accuracies_by_threshold, accuracy_at_thr
import matplotlib.pyplot as plt


# this model must have streaming clf as input
class GeoTransformBaseDirichletAlphasSaved(GeoTransformBase):
    def __init__(self, classifier: StreamingTransformationsDeepHits,
        transformer: AbstractTransformer, results_folder_name=None,
        name='GeoTransform_Base_Dirichlet_Alphas_Saved'):
        super().__init__(classifier, transformer, results_folder_name, name)
        self.dirichlet_alphas = self._initialize_dirichlet_alphas()
        self.prediction_threshold = 27380.5240241269 # random

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

    def _update_prediction_threshold(self, x_validation, verbose):
        transform_batch_size = 512
        evaluation_batch_size = 1024
        print('Calculating prediction threshold...')
        dirichlet_scores_validation = self.predict_dirichlet_score(
            x_validation, transform_batch_size, evaluation_batch_size, verbose)
        self.prediction_threshold = np.percentile(
            dirichlet_scores_validation, 100 - self.percentile)


    def fit(self, x_train, epochs, x_validation=None, batch_size=128,
        iterations_to_validate=None, patience=None, verbose=True):
        print_manager = PrintManager().verbose_printing(verbose)
        if epochs is None:
            epochs = self._get_original_paper_epochs()
            patience = int(1e100)
        self.classifier.fit(
            x_train, epochs, x_validation, batch_size, iterations_to_validate,
            patience, verbose)
        self._update_dirichlet_alphas(x_train, verbose)
        if x_validation is None:
            x_validation = x_train
        self._update_prediction_threshold(x_validation, verbose)
        self.save_model(self.classifier.best_model_weights_path)
        print_manager.close()

    # TODO: avoid apply_all_transforms at once
    def _predict_matrix_probabilities(self, x_data, transform_batch_size=512,
        predict_batch_size=1024):
        return

    def predict_dirichlet_score(self, x_eval, transform_batch_size=512,
        predict_batch_size=1024, verbose=True):
        print_manager = PrintManager().verbose_printing(verbose)
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
        print_manager.close()
        return dirichlet_scores

    def _get_metrics_dict(self, scores, labels):
        divided_scores, divided_labels = self.\
            _assign_inlier_outlier_sections_to_scores_and_labels(scores, labels)
        fpr, tpr, roc_thresholds = roc_curve(divided_labels, divided_scores)
        roc_auc = auc(fpr, tpr)
        accuracies = accuracies_by_threshold(labels, scores, roc_thresholds)
        # 100-percentile is necesary because normal data is at the right of
        # anormal
        thr = self.prediction_threshold
        acc_at_percentil = accuracy_at_thr(labels, scores, thr)
        # pr curve where "normal" is the positive class
        precision_norm, recall_norm, pr_thresholds_norm = \
            precision_recall_curve(divided_labels, divided_scores)
        pr_auc_norm = auc(recall_norm, precision_norm)
        # pr curve where "anomaly" is the positive class
        precision_anom, recall_anom, pr_thresholds_anom = \
            precision_recall_curve(divided_labels, -divided_scores, pos_label=0)
        pr_auc_anom = auc(recall_anom, precision_anom)
        metrics_dict = {'scores': scores, 'labels': labels,
                        'clf': (scores > thr) * 1,
                        'fpr': fpr,
                        'tpr': tpr, 'roc_thresholds': roc_thresholds,
                        'roc_auc': roc_auc,
                        'precision_norm': precision_norm,
                        'recall_norm': recall_norm,
                        'pr_thresholds_norm': pr_thresholds_norm,
                        'pr_auc_norm': pr_auc_norm,
                        'precision_anom': precision_anom,
                        'recall_anom': recall_anom,
                        'pr_thresholds_anom': pr_thresholds_anom,
                        'pr_auc_anom': pr_auc_anom,
                        'accuracies': accuracies,
                        'max_accuracy': np.max(accuracies),
                        'acc_at_percentil': acc_at_percentil}
        return metrics_dict

    def _save_histogram(self, score_metric_dict, score_name, dataset_name,
        class_name, save_histogram=False):
        if not save_histogram:
            return
        auc_roc = score_metric_dict['roc_auc']
        accuracies = score_metric_dict['accuracies']
        scores = score_metric_dict['scores']
        labels = score_metric_dict['labels']
        thresholds = score_metric_dict['roc_thresholds']
        accuracy_at_percentile = score_metric_dict['acc_at_percentil']
        inliers_scores = scores[labels == 1]
        outliers_scores = scores[labels != 1]
        min_score = np.min(scores)
        max_score = np.max(scores)
        thr_percentile = self.prediction_threshold
        fig = plt.figure(figsize=(8, 6))
        ax_hist = fig.add_subplot(111)
        ax_hist.set_title(
            'AUC_ROC: %.2f%%, BEST ACC: %.2f%%' % (
                auc_roc * 100, np.max(accuracies) * 100))
        ax_acc = ax_hist.twinx()
        ax_hist.hist(inliers_scores, 300, alpha=0.5,
                             label='inlier', range=[min_score, max_score])
        ax_hist.hist(outliers_scores, 300, alpha=0.5,
                             label='outlier', range=[min_score, max_score])
        ax_hist.set_yscale('log')
        _, max_ = ax_hist.set_ylim()
        ax_hist.set_ylabel('Counts', fontsize=12)
        ax_hist.set_xlabel(score_name, fontsize=12)
        # acc plot
        ax_acc.set_ylim([0.5, 1.0])
        ax_acc.yaxis.set_ticks(np.arange(0.5, 1.05, 0.05))
        ax_acc.set_ylabel('Accuracy', fontsize=12)
        ax_acc.plot(thresholds, accuracies, lw=2,
                               label='Accuracy by\nthresholds',
                               color='black')
        ax_hist.plot([thr_percentile, thr_percentile],
                                      [0, max_],
                                      'k--',
                                      label='thr percentil %i on %s' % (
                                          self.percentile, dataset_name))
        ax_hist.text(thr_percentile,
                     max_ * 0.6,
                     'Acc: {:.2f}%'.format(accuracy_at_percentile * 100))
        ax_acc.grid(ls='--')
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1),
                   bbox_transform=ax_hist.transAxes)
        results_name = self._get_score_result_name(score_name, dataset_name,
                                                   class_name)
        utils.check_paths(self.model_results_path)
        fig.savefig(
            os.path.join(
                self.model_results_path, '%s_hist_thr_acc.png' % results_name),
            bbox_inches='tight')
        plt.close()

    def evaluate(self, x_eval, y_eval, dataset_name,
        class_name='inlier', transform_batch_size=512,
        evaluation_batch_size=1024, save_metrics=False,
        additional_score_save_path_list=None, save_histogram=False,
        get_auroc_acc_only=False, verbose=True):
        print_manager = PrintManager().verbose_printing(verbose)
        print('\nEvaluating model...')
        start_time = time.time()
        dirichlet_scores_eval = self.predict_dirichlet_score(
            x_eval, transform_batch_size, evaluation_batch_size, verbose)
        metrics = self._get_metrics_dict(
            dirichlet_scores_eval, y_eval)
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
        print_manager.close()
        return metrics

    def save_model(self, path):
        # Include alphas saving
        folder_path = os.path.dirname(path)
        dirichlet_mle_alphas_path = os.path.join(
            folder_path, 'dirichlet_mle_alphas.npy')
        np.save(dirichlet_mle_alphas_path, self.dirichlet_alphas)
        prediction_threshold_path = os.path.join(
            folder_path, 'prediction_threshold.npy')
        np.save(prediction_threshold_path, self.prediction_threshold)
        self.classifier.save_weights(path)

    def load_model(self, path_checkpoints, by_name=False):
        # Include alphas saving
        folder_path = os.path.dirname(path_checkpoints)
        self.dirichlet_alphas = np.load(
            os.path.join(folder_path, 'dirichlet_mle_alphas.npy'))
        self.prediction_threshold = np.load(
            os.path.join(folder_path, 'prediction_threshold.npy'))
        self.classifier.load_weights(path_checkpoints, by_name=by_name)

    def predict(self, x_eval, x_validation=None,
        transform_batch_size=512, evaluation_batch_size=1024, verbose=False):
        dirichlet_scores_eval = self.predict_dirichlet_score(
            x_eval, transform_batch_size, evaluation_batch_size, verbose)
        predictions = (dirichlet_scores_eval > self.prediction_threshold) * 1
        return predictions


def get_best_ztf_tuples(add_zeros=True):
    tuples = [(0, 0, 0, 0, 0, 0), (0, -8, -8, 1, 0, 0), (0, -8, -8, 3, 0, 0),
              (0, 8, 0, 0, 1, 1), (0, 8, -8, 0, 1, 0), (0, -8, 0, 0, 1, 1),
              (0, 0, 8, 0, 1, 1), (0, -8, -8, 2, 0, 0), (0, -8, 8, 0, 0, 1),
              (0, 0, 0, 0, 1, 1), (0, 0, 0, 0, 1, 0), (0, 0, -8, 0, 1, 1),
              (0, 8, 8, 0, 0, 1), (0, -8, -8, 0, 0, 1), (0, 0, -8, 0, 0, 0),
              (0, 8, -8, 0, 1, 1), (0, -8, -8, 0, 0, 0)]
    if add_zeros:
        for i, tup in enumerate(tuples):
            tuples[i] = tuple(list(tup) + [0, 0])
    return tuples


def get_best_9_tuples():
    tuples = [(False, 0, 0, 0), (False, 0, -8, 0),
              (False, 0, 8, 0), (False, -8, 0, 0),
              (False, -8, -8, 0), (False, -8, 8, 0),
              (False, 8, 0, 0), (False, 8, -8, 0),
              (False, 8, 8, 0)]
    for i, tup in enumerate(tuples):
        tuples[i] = tuple(list(tup) + [0, 0, 0, 0])
    return tuples


def get_best_hits_tuples(add_zeros=True):
    tuples = [(0, 0, 0, 0, 0, 0), (0, 0, -8, 0, 0, 0), (0, 0, -8, 0, 1, 0),
              (0, 8, 0, 0, 0, 1), (0, -8, 8, 0, 0, 1), (0, -8, -8, 0, 0, 1),
              (0, 8, -8, 0, 1, 1), (0, -8, 8, 0, 1, 0), (0, -8, -8, 3, 0, 0),
              (0, -8, -8, 0, 1, 1), (0, -8, -8, 1, 0, 0), (0, 8, 8, 0, 0, 1),
              (0, -8, 0, 0, 0, 1), (0, 8, 8, 0, 1, 1), (0, 8, 0, 0, 1, 1),
              (0, 8, -8, 0, 0, 1), (0, 8, -8, 0, 1, 0), (0, -8, -8, 2, 0, 0),
              (0, -8, 8, 0, 1, 1), (0, -8, -8, 0, 1, 0), (0, -8, 0, 0, 1, 1)]
    if add_zeros:
        for i, tup in enumerate(tuples):
            tuples[i] = tuple(list(tup) + [0, 0])
    return tuples


if __name__ == '__main__':
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from modules.data_loaders.ztf_small_outlier_loader import \
        ZTFSmallOutlierLoader
    from parameters import loader_keys
    from modules.geometric_transform. \
        streaming_transformers.transformer_ranking import RankingTransformer
    from modules.networks.streaming_network. \
        streaming_transformations_wide_resnet import \
        StreamingTransformationsWideResnet
    import matplotlib

    matplotlib.use('Agg')

    EPOCHS = 1#1000
    # 000
    ITERATIONS_TO_VALIDATE = 2#0
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
    # outlier_loader = ztf_loader
    outlier_loader = hits_loader

    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = outlier_loader.get_outlier_detection_datasets()
    transformer = RankingTransformer()
    transformer.set_transformations_to_perform(get_best_hits_tuples())
    # transformer.set_transformations_to_perform(get_best_9_tuples())
    # transformer.set_transformations_to_perform(get_best_tuples())
    # clf = StreamingTransformationsDeepHits(transformer)
    clf = StreamingTransformationsWideResnet(x_train.shape[:-1],
                                             transformer)
    model = GeoTransformBaseDirichletAlphasSaved(
        classifier=clf, transformer=transformer,
        results_folder_name='test_base')
    # model.predict(x_test, x_val)
    model.fit(
        x_train, epochs=EPOCHS, x_validation=x_val,
        iterations_to_validate=ITERATIONS_TO_VALIDATE, patience=PATIENCE,
        verbose=VERBOSE)
    model.evaluate(
        x_test, y_test, outlier_loader.name, 'real', save_metrics=True,
        save_histogram=True, get_auroc_acc_only=True, verbose=VERBOSE)
    model_results_folder = model.model_results_path
    model_weights_path = model.classifier.best_model_weights_path
    del model
    del clf
    # clf = StreamingTransformationsDeepHits(transformer)
    clf = StreamingTransformationsWideResnet(x_train.shape[:-1],
                                             transformer)
    model = GeoTransformBaseDirichletAlphasSaved(
        classifier=clf, transformer=transformer,
        results_folder_name='test_base')
    model.set_model_results_path(model_results_folder)
    model.load_model(model_weights_path)
    model.evaluate(
        x_test, y_test, outlier_loader.name, 'real', save_metrics=True,
        save_histogram=True, get_auroc_acc_only=True, verbose=VERBOSE)
    print(np.mean(model.predict(x_test) == y_test))
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
