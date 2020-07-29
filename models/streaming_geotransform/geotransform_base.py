"""
First attempt (train_step_tf2 like) to build geometric trasnformer for outlier detection in tf2
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)
import tensorflow as tf
# from modules.networks.wide_residual_network import WideResidualNetwork
from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from parameters import general_keys
import numpy as np
from modules import dirichlet_utils, utils
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from modules.metrics import accuracies_by_threshold, accuracy_at_thr
import pprint
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from sklearn.svm import OneClassSVM
import time
from modules.print_manager import PrintManager
from modules.networks.streaming_network.streaming_transformations_deep_hits\
    import StreamingTransformationsDeepHits


class GeoTransformBase(tf.keras.Model):
    def __init__(self, classifier: StreamingTransformationsDeepHits,
        transformer: AbstractTransformer, results_folder_name=None,
        name='GeoTransform_Base'):
        super().__init__(name=name)
        self.classifier = classifier
        self.main_model_path = self._create_model_paths(
            results_folder_name)
        self.classifier._set_model_paths(self.main_model_path)
        self.print_manager = PrintManager()
        self.transformer = transformer
        self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.percentile = 97.73

    def classifier_call(self, input_tensor, training=False):
        return self.classifier(input_tensor, training)

    def _create_model_paths(self, results_folder_name, model_name):
        if results_folder_name is None:
            results_folder_name = self.name
            results_folder_path = results_folder_name
        else:
            results_folder_path = os.path.join(
                PROJECT_PATH, 'results', results_folder_name,
                '%s_%s' % (self.name, self.date))
        utils.check_path(results_folder_path)
        return results_folder_path

    def _get_original_paper_epochs(self):
        return int(np.ceil(200 / self.transformer.n_transforms))

    def fit_classifier(self, x_train, epochs, x_validation=None, batch_size=128,
        iterations_to_validate=None, patience=None, verbose=True):
        if epochs is None:
            epochs = self._get_original_paper_epochs()
            patience = int(1e100)
        self.classifier.fit(
            x_train, epochs, x_validation, batch_size, iterations_to_validate,
            patience, verbose)

    def predict_dirichlet_score(self, x_train, x_eval,
        transform_batch_size=512, predict_batch_size=1024, verbose=True,
        **kwargs):
        self.print_manager.verbose_printing(verbose)
        _, diri_scores = self.predict_matrix_and_dirichlet_score(
            x_train, x_eval, transform_batch_size, predict_batch_size, verbose,
            **kwargs)
        self.print_manager.close()
        return diri_scores

    # TODO: avoid apply_all_transforms at once
    def _predict_matrix_probabilities(self, x_data, transform_batch_size=512,
        predict_batch_size=1024):
        n_transforms = self.transformer.n_transforms
        x_transformed, y_transformed = self.transformer.apply_all_transforms(
            x_data, transform_batch_size)
        start_time = time.time()
        x_pred = self.classifier.predict(x_transformed,
                                         batch_size=predict_batch_size)
        # get actual length if all transformations applied on model input
        matrix_scores = np.zeros((len(x_data), n_transforms, n_transforms))
        for t_ind in range(n_transforms):
            ind_x_pred_equal_to_t_ind = np.where(y_transformed == t_ind)[0]
            matrix_scores[:, :, t_ind] += x_pred[ind_x_pred_equal_to_t_ind]
        print("Matrix probabilities time: %s" % utils.timer(
                start_time, time.time()))
        return matrix_scores

    # TODO: avoid apply_all_transforms at once
    # TODO: instead of saving matrix_train_score save dirichlet params
    def predict_matrix_and_dirichlet_score(self, x_train, x_eval,
        transform_batch_size=512, predict_batch_size=1024, verbose=True):
        self.print_manager.verbose_printing(verbose)
        n_transforms = self.transformer.n_transforms
        if self.matrix_scores_train is None:
            print("Calculating matrix probabilities for training set...")
            self.matrix_scores_train = self._predict_matrix_probabilities(
                x_train, transform_batch_size, predict_batch_size)
        # TODO: see effect of this del x_train
        del x_train
        # TODO: avoid calculation of whole eval matrix, if there ire too many
        #  transformations, matrices will be bigger than actual data!
        print("Calculating matrix probabilities for evaluation set...")
        matrix_scores_eval = self._predict_matrix_probabilities(
            x_eval, transform_batch_size, predict_batch_size)
        # get actual length if all transformations applied on model input
        diri_scores = np.zeros(len(x_eval))
        # TODO: see effect of this del x_eval
        del x_eval
        for t_ind in tqdm(range(n_transforms), disable=not verbose):
            observed_dirichlet = self.matrix_scores_train[:, :, t_ind]
            x_eval_p = matrix_scores_eval[:, :, t_ind]
            diri_scores += dirichlet_utils.dirichlet_score(
                observed_dirichlet, x_eval_p)
            assert np.isfinite(diri_scores).all()
        diri_scores /= n_transforms
        self.print_manager.close()
        return matrix_scores_eval, diri_scores

    # TODO: HERE!!!
    def get_scores_dict(self, x_train, x_eval,
        transform_batch_size=512, predict_batch_size=1024, verbose=True,
        **kwargs):
        matrix_scores, diri_scores = self.predict_matrix_and_dirichlet_score(
            x_train, x_eval, transform_batch_size, predict_batch_size,
            verbose, **kwargs)
        matrix_scores = matrix_scores / self.transformer.n_transforms
        scores_dict = {
            general_keys.DIRICHLET: diri_scores,
            # general_keys.MATRIX_TRACE: np.trace(matrix_scores, axis1=1, axis2=2),
            # general_keys.ENTROPY: -1 * score_functions.get_entropy(matrix_scores),
            # general_keys.CROSS_ENTROPY: -1 * score_functions.get_xH(self.transformer,
            #                                                         matrix_scores),
            # general_keys.MUTUAL_INFORMATION: score_functions.get_xy_mutual_info(
            #     matrix_scores)
        }
        return scores_dict

    def _process_data_for_svm(self, data):
        y_proy = data.sum(axis=1)
        x_proy = data.sum(axis=2)
        xy_proy = np.concatenate([y_proy, x_proy], axis=-1)
        return xy_proy

    # TODO: refactor, too long and include keys
    def get_metrics_save_path(self, score_name, dataset_name, class_name):
        results_name = self._get_score_result_name(score_name, dataset_name,
                                                   class_name)
        results_file_name = '{}.npz'.format(results_name)
        all_results_folder = os.path.join(self.main_model_path,
                                          'all_metric_files',
                                          dataset_name)
        utils.check_paths(all_results_folder)
        results_file_path = os.path.join(all_results_folder, results_file_name)
        return results_file_path

    # TODO: refactor, too long
    def evaluate_od(self, x_train, x_eval, y_eval, dataset_name, class_name,
        x_validation=None, transform_batch_size=512, predict_batch_size=1000,
        additional_score_save_path_list=None, save_hist_folder_path=None,
        verbose=True,
        **kwargs):
        self.print_manager.verbose_printing(verbose)
        # TODO: avoid doing this!! need refctoring, but avoid repreedict of training
        self.matrix_scores_train = None
        print('evaluating')
        if x_validation is None:
            x_validation = x_eval
        # print('start eval')
        eval_scores_dict = self.get_scores_dict(
            x_train, x_eval, transform_batch_size, predict_batch_size, verbose,
            **kwargs)
        # print('start val')
        validation_scores_dict = self.get_scores_dict(
            x_train, x_validation, transform_batch_size, predict_batch_size,
            verbose, **kwargs)
        # del self.matrix_scores_train
        # print('start metric')
        metrics_of_each_score = {}
        for score_name, scores_value in eval_scores_dict.items():
            metrics_save_path = self.get_metrics_save_path(score_name,
                                                           dataset_name,
                                                           class_name)
            metrics_of_each_score[score_name] = self.get_metrics_dict(
                scores_value, validation_scores_dict[score_name], y_eval,
                metrics_save_path)
            self._save_on_additional_paths(
                additional_score_save_path_list, metrics_save_path,
                metrics_of_each_score[score_name])
            self._save_histogram(metrics_of_each_score[score_name], score_name,
                                 dataset_name, class_name,
                                 save_hist_folder_path)
        # print('hola')
        self.print_manager.close()
        return metrics_of_each_score

    def _get_score_result_name(self, score_name, dataset_name,
        class_name):
        model_score_name = ('%s_%s' % (self.name, score_name)).replace("_", "-")
        dataset_plus_transformer_name = (
            '%s_%s' % (dataset_name, self.transformer.name)).replace("_", "-")
        results_name = '{}_{}_{}_{}'.format(
            dataset_plus_transformer_name, model_score_name, class_name,
            self.date)
        return results_name

    def _save_histogram(self, score_metric_dict, score_name, dataset_name,
        class_name, save_folder_path=None):
        if save_folder_path is None:
            return
        # TODO: refactor usage of percentile, include it in metrics and
        #  get it from key
        # percentile = 95.46
        scores_val = score_metric_dict['scores_val']
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
        thr_percentile = np.percentile(scores_val, 100 - self.percentile)
        fig = plt.figure(figsize=(8, 6))
        ax_hist = fig.add_subplot(111)
        ax_hist.set_title(
            'AUC_ROC: %.2f%%, BEST ACC: %.2f%%' % (
                auc_roc * 100, np.max(accuracies) * 100))
        ax_acc = ax_hist.twinx()
        hist1 = ax_hist.hist(inliers_scores, 300, alpha=0.5,
                             label='inlier', range=[min_score, max_score])
        hist2 = ax_hist.hist(outliers_scores, 300, alpha=0.5,
                             label='outlier', range=[min_score, max_score])
        ax_hist.set_yscale('log')
        _, max_ = ax_hist.set_ylim()
        ax_hist.set_ylabel('Counts', fontsize=12)
        ax_hist.set_xlabel(score_name, fontsize=12)
        # acc plot
        ax_acc.set_ylim([0.5, 1.0])
        ax_acc.yaxis.set_ticks(np.arange(0.5, 1.05, 0.05))
        ax_acc.set_ylabel('Accuracy', fontsize=12)
        acc_plot = ax_acc.plot(thresholds, accuracies, lw=2,
                               label='Accuracy by\nthresholds',
                               color='black')
        percentil_plot = ax_hist.plot([thr_percentile, thr_percentile],
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
        fig.savefig(
            os.path.join(save_folder_path,
                         '%s_hist_thr_acc.png' % results_name),
            bbox_inches='tight')
        plt.close()

    # TODO: refactor to avoid usage of metrics_save_path,
    # this should be additional_paths_list and data
    def _save_on_additional_paths(self, additional_paths_list: list,
        metrics_save_path: str, metrics_dict: dict):
        if additional_paths_list is None:
            return
        if not isinstance(additional_paths_list, list):
            additional_paths_list = [additional_paths_list]
        for path in additional_paths_list:
            metric_file_name = os.path.basename(metrics_save_path)
            additional_save_path = os.path.join(path, metric_file_name)
            np.savez_compressed(additional_save_path, **metrics_dict)

    def get_metrics_dict(self, scores, scores_val, labels,
        save_file_path=None):  # ,
        # percentile=95.46):
        scores = scores.flatten()
        labels = labels.flatten()
        scores_pos = scores[labels == 1]
        scores_neg = scores[labels != 1]
        truth = np.concatenate(
            (np.zeros_like(scores_neg), np.ones_like(scores_pos)))
        preds = np.concatenate((scores_neg, scores_pos))
        fpr, tpr, roc_thresholds = roc_curve(truth, preds)
        roc_auc = auc(fpr, tpr)
        accuracies = accuracies_by_threshold(labels, scores, roc_thresholds)
        # 100-percentile is necesary because normal data is at the right of anormal
        thr = np.percentile(scores_val, 100 - self.percentile)
        acc_at_percentil = accuracy_at_thr(labels, scores, thr)
        # pr curve where "normal" is the positive class
        precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(
            truth, preds)
        pr_auc_norm = auc(recall_norm, precision_norm)
        # pr curve where "anomaly" is the positive class
        precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(
            truth, -preds, pos_label=0)
        pr_auc_anom = auc(recall_anom, precision_anom)
        metrics_dict = {'scores': scores, 'labels': labels,
                        'clf': (scores > thr) * 1,
                        'scores_val': scores_val, 'fpr': fpr,
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
        if save_file_path is not None:
            # TODO: check if **metrics_dict works
            np.savez_compressed(save_file_path, **metrics_dict)
        else:
            pprint.pprint((metrics_dict))
        return metrics_dict


if __name__ == '__main__':
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from parameters import loader_keys
    from modules.geometric_transform.streaming_transformers.transformer_ranking import RankingTransformer


    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # data_loader_params = {
    #   loader_keys.DATA_PATH: os.path.join(
    #       PROJECT_PATH, '../datasets/ztf_v1_bogus_added.pkl'),
    #   loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    #   loader_keys.USED_CHANNELS: [0, 1, 2],
    #   loader_keys.CROP_SIZE: 21,
    #   general_keys.RANDOM_SEED: 42,
    #   loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    # }
    # outlier_loader = ZTFOutlierLoader(data_loader_params)
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 10000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],  # [2],  #
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    outlier_loader = HiTSOutlierLoader(hits_params)
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = outlier_loader.get_outlier_detection_datasets()
    transformer = RankingTransformer()
    clf = DeepHits(n_classes=transformer.n_transforms)
    model = GeoTransformBase(
        classifier=clf, data_loader=outlier_loader, transformer=transformer)
    model.fit(x_train, x_val, epochs=10)