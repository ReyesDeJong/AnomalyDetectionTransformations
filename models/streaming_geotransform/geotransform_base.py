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

# this model must have streaming clf as input
class GeoTransformBase(tf.keras.Model):
    def __init__(self, classifier: StreamingTransformationsDeepHits,
        transformer: AbstractTransformer, results_folder_name=None,
        name='GeoTransform_Base'):
        super().__init__(name=name)
        self.classifier = classifier
        self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_results_path = self._create_model_paths(
            results_folder_name)
        self.classifier._set_model_results_paths(self.model_results_path)
        self.transformer = transformer
        self.percentile = 97.73
        self.matrix_scores_train = None

    def set_model_results_path(self, path):
        self.model_results_path = path

    def classifier_call(self, input_tensor, training=False):
        return self.classifier(input_tensor, training)

    def _create_model_paths(self, results_folder_name):
        if results_folder_name is None:
            results_folder_name = self.name
            results_folder_path = results_folder_name
        else:
            results_folder_path = os.path.join(
                PROJECT_PATH, 'results', results_folder_name,
                '%s_%s_%s' % (self.name, self.classifier.name, self.date))
        # utils.check_path(results_folder_path)
        return results_folder_path

    def _get_original_paper_epochs(self):
        return int(np.ceil(200 / self.transformer.n_transforms))

    def fit(self, x_train, epochs, x_validation=None, batch_size=128,
        iterations_to_validate=None, patience=None, verbose=True):
        if epochs is None:
            epochs = self._get_original_paper_epochs()
            patience = int(1e100)
        self.classifier.fit(
            x_train, epochs, x_validation, batch_size, iterations_to_validate,
            patience, verbose)

    # TODO: avoid apply_all_transforms at once
    def _predict_matrix_probabilities(self, x_data, transform_batch_size=512,
        predict_batch_size=1024):
        start_time = time.time()
        n_transforms = self.transformer.n_transforms
        x_transformed, y_transformed = self.transformer.apply_all_transforms(
            x_data, transform_batch_size)
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
    def predict_dirichlet_score(self, x_train, x_eval,
        transform_batch_size=512, predict_batch_size=1024, verbose=True):
        print_manager = PrintManager().verbose_printing(verbose)
        n_transforms = self.transformer.n_transforms
        if self.matrix_scores_train is None:
            print("Calculating matrix probabilities for training set...")
            self.matrix_scores_train = self._predict_matrix_probabilities(
                x_train, transform_batch_size, predict_batch_size)
        # TODO: see effect of this del x_train
        del x_train
        # TODO: avoid calculation of whole eval matrix, if there ire too many
        #  transformations, matrices will be bigger than actual data!
        print("Calculating matrix probabilities...")
        matrix_scores_eval = self._predict_matrix_probabilities(
            x_eval, transform_batch_size, predict_batch_size)
        # get actual length if all transformations applied on model input
        dirichlet_scores = np.zeros(len(x_eval))
        # TODO: see effect of this del x_eval
        del x_eval
        # TODO: do parallel of this
        print('Calculating dirichlet scores...')
        for t_ind in tqdm(range(n_transforms), disable=not verbose):
            observed_dirichlet = self.matrix_scores_train[:, :, t_ind]
            x_eval_p = matrix_scores_eval[:, :, t_ind]
            dirichlet_scores += dirichlet_utils.dirichlet_score(
                observed_dirichlet, x_eval_p)
            assert np.isfinite(dirichlet_scores).all()
        dirichlet_scores /= n_transforms
        print_manager.close()
        return dirichlet_scores

    def get_metrics_save_path(
        self, score_name, dataset_name, class_name):
        results_name = self._get_score_result_name(score_name, dataset_name,
                                                   class_name)
        results_file_name = '{}.npz'.format(results_name)
        metrics_folder_path = os.path.join(
            self.model_results_path)
        utils.check_paths(metrics_folder_path)
        results_file_path = os.path.join(metrics_folder_path, results_file_name)
        return results_file_path

    def _get_score_result_name(
        self, score_name, dataset_name, class_name):
        model_score_name = ('%s-%s' % (self.name, score_name)).replace("_", "-")
        dataset_plus_transformer_name = (
            '%s-%s' % (dataset_name, self.transformer.name)).replace("_", "-")
        results_name = '{}_{}_{}_{}'.format(
            dataset_plus_transformer_name, model_score_name, class_name,
            self.date)
        return results_name

    def evaluate(self, x_train, x_eval, y_eval, dataset_name,
        class_name='inlier', x_validation=None, transform_batch_size=512,
        evaluation_batch_size=1024, save_metrics=False,
        additional_score_save_path_list=None, save_histogram=False,
        get_auroc_acc_only=False, verbose=True):
        print_manager = PrintManager().verbose_printing(verbose)
        print('\nEvaluating model...')
        start_time = time.time()
        # validatioon is ussed to set accuracy thresholds
        if x_validation is None:
            x_validation = x_eval
        dirichlet_scores_eval = self.predict_dirichlet_score(
            x_train, x_eval, transform_batch_size, evaluation_batch_size,
            verbose)
        dirichlet_scores_validation = self.predict_dirichlet_score(
            x_train, x_validation, transform_batch_size, evaluation_batch_size,
            verbose)
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
        print_manager.close()
        return metrics

    def _print_final_metrics(
        self, metrics: dict, keys_to_keep: list):
        message = '\nEvaluation metrics:'
        for key_i in keys_to_keep:
            message += ' %s %.6f,' % (key_i, metrics[key_i])
        message = message[:-1]
        print(message)

    def _filter_metrics_to_return(
        self, metrics: dict, get_auroc_acc_only, keys_to_keep: list):
        if get_auroc_acc_only:
            filtered_metrics = {
                your_key: metrics[your_key] for your_key in keys_to_keep}
            return filtered_metrics
        return metrics

    def _save_histogram(self, score_metric_dict, score_name, dataset_name,
        class_name, save_histogram=False):
        if not save_histogram:
            return
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

    def _save_histogram_log_log(self, score_metric_dict, score_name,
        dataset_name, class_name, save_histogram=False):
        if not save_histogram:
            return
        scores_val = score_metric_dict['scores_val']
        auc_roc = score_metric_dict['roc_auc']
        accuracies = score_metric_dict['accuracies']
        scores = score_metric_dict['scores']
        labels = score_metric_dict['labels']
        thresholds = score_metric_dict['roc_thresholds']
        accuracy_at_percentile = score_metric_dict['acc_at_percentil']
        inliers_scores = -scores[labels == 1]
        outliers_scores = -scores[labels != 1]
        min_score = np.min(-scores)
        max_score = np.max(-scores)
        thr_percentile = -np.percentile(scores_val, 100 - self.percentile)
        fig = plt.figure(figsize=(8, 6))
        ax_hist = fig.add_subplot(111)
        ax_hist.set_yscale('log')
        ax_hist.set_xscale('log')
        ax_hist.set_title(
            'AUC_ROC: %.2f%%, BEST ACC: %.2f%%' % (
                auc_roc * 100, np.max(accuracies) * 100))
        ax_acc = ax_hist.twinx()
        ax_hist.hist(inliers_scores, 300, alpha=0.5,
                             label='inlier', range=[min_score, max_score])
        ax_hist.hist(outliers_scores, 300, alpha=0.5,
                             label='outlier', range=[min_score, max_score])
        _, max_ = ax_hist.set_ylim()
        ax_hist.set_ylabel('Counts', fontsize=12)
        ax_hist.set_xlabel(score_name, fontsize=12)
        # acc plot
        ax_acc.set_ylim([0.5, 1.0])
        ax_acc.yaxis.set_ticks(np.arange(0.5, 1.05, 0.05))
        ax_acc.set_ylabel('Accuracy', fontsize=12)
        ax_acc.plot(-thresholds, accuracies, lw=2,
                               label='Accuracy by\nthresholds',
                               color='black')
        ax_hist.plot([thr_percentile, thr_percentile],
                                      [0, max_],
                                      'k--',
                                      label='thr percentil %i on %s' % (
                                          self.percentile, dataset_name))
        ax_hist.text(thr_percentile,
                     10, 'Acc: {:.2f}%'.format(accuracy_at_percentile * 100))
        ax_acc.grid(ls='--')
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1),
                   bbox_transform=ax_hist.transAxes)
        results_name = self._get_score_result_name(score_name, dataset_name,
                                                   class_name)
        fig.savefig(
            os.path.join(
                self.model_results_path, '%s_hist_thr_acc.png' % results_name),
            bbox_inches='tight')
        plt.close()

    def _save_metrics_on_paths(self, additional_paths_list: list,
        metrics_save_path: str, metrics_dict: dict, save_metrics):
        if not save_metrics:
            return
        np.savez_compressed(metrics_save_path, **metrics_dict)
        if additional_paths_list is None:
            return
        if not isinstance(additional_paths_list, list):
            additional_paths_list = [additional_paths_list]
        for path in additional_paths_list:
            metric_file_name = os.path.basename(metrics_save_path)
            additional_save_path = os.path.join(path, metric_file_name)
            np.savez_compressed(additional_save_path, **metrics_dict)

    def _assign_inlier_outlier_sections_to_scores_and_labels(
        self, scores, labels):
        scores = scores.flatten()
        labels = labels.flatten()
        # inliers
        scores_pos = scores[labels == 1]
        # outliers
        scores_neg = scores[labels != 1]
        divided_labels = np.concatenate(
            (np.zeros_like(scores_neg), np.ones_like(scores_pos)))
        divided_scores = np.concatenate((scores_neg, scores_pos))
        return divided_scores, divided_labels

    def _get_metrics_dict(self, scores, scores_val, labels):
        divided_scores, divided_labels = self.\
            _assign_inlier_outlier_sections_to_scores_and_labels(scores, labels)
        fpr, tpr, roc_thresholds = roc_curve(divided_labels, divided_scores)
        roc_auc = auc(fpr, tpr)
        accuracies = accuracies_by_threshold(labels, scores, roc_thresholds)
        # 100-percentile is necesary because normal data is at the right of
        # anormal
        thr = np.percentile(scores_val, 100 - self.percentile)
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
        return metrics_dict

    def save_weights(self, path):
        self.classifier.save_weights(path)

    def load_weights(self, path, by_name=False):
        self.classifier.load_weights(path, by_name=by_name)

    def predict(self, x_train, x_eval, x_validation=None,
        transform_batch_size=512, evaluation_batch_size=1024, verbose=False):
        if x_validation is None:
            x_validation = x_eval
        dirichlet_scores_eval = self.predict_dirichlet_score(
            x_train, x_eval, transform_batch_size, evaluation_batch_size,
            verbose)
        dirichlet_scores_validation = self.predict_dirichlet_score(
            x_train, x_validation, transform_batch_size, evaluation_batch_size,
            verbose)
        thr = np.percentile(dirichlet_scores_validation, 100 - self.percentile)
        predictions = (dirichlet_scores_eval > thr) * 1
        return predictions

if __name__ == '__main__':
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from modules.data_loaders.ztf_small_outlier_loader import \
        ZTFSmallOutlierLoader
    from parameters import loader_keys
    from modules.geometric_transform.\
        streaming_transformers.transformer_ranking import RankingTransformer
    import matplotlib
    matplotlib.use('Agg')

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
    model = GeoTransformBase(
        classifier=clf, transformer=transformer,
        results_folder_name='test_base')
    model.fit(
        x_train, epochs=EPOCHS, x_validation=x_val,
        iterations_to_validate=ITERATIONS_TO_VALIDATE, patience=PATIENCE,
        verbose=VERBOSE)
    model.evaluate(
        x_train, x_test, y_test, outlier_loader.name, 'real', x_val,
        save_metrics=True, save_histogram=True, get_auroc_acc_only=True,
        verbose=VERBOSE)
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
    #     x_train, x_test, y_test, outlier_loader.name, 'real', x_val,
    #     save_metrics=False, save_histogram=True, get_auroc_acc_only=True,
    #     verbose=VERBOSE)
