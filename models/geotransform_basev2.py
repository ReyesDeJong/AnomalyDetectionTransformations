"""
a base geotransform both for streaming and normal model, that is defined with
classifier.

It aplplies transformations in fit method, need mods if streaming is passed

WARNING! to correctly update prediction thr, pass validation when original setup is used,
no validation data is used for thr generation

taken from models/streaming_geotransform/geotransform_base.py and
geotranform_base_dirichlet_alphas_save.py
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

import tensorflow as tf
from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
from parameters import general_keys
import numpy as np
from modules import dirichlet_utils, utils
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from modules.metrics import accuracies_by_threshold, accuracy_at_thr, \
    precision_at_thr, recall_at_thr
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from modules.print_manager import PrintManager
from modules.networks.deep_hitsv2 import DeepHitsv2
import warnings

# this model must have streaming clf as input
class GeoTransformBasev2(tf.keras.Model):
    def __init__(self, classifier: DeepHitsv2,
        transformer: AbstractTransformer, results_folder_name=None,
        name='GeoTransform_Basev2'):
        super().__init__(name=name)
        self.classifier = classifier
        self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_results_path = self._create_model_paths(
            results_folder_name)
        self.classifier._set_model_results_paths(self.model_results_path)
        self.transformer = transformer
        self.percentile = 97.73
        self.matrix_scores_train = None

        self.dirichlet_alphas = self._initialize_dirichlet_alphas()
        self.prediction_threshold = 27380.5240241269  # random

    def _initialize_dirichlet_alphas(self):
        n_transforms = self.transformer.n_transforms
        return np.random.normal(
            size=(n_transforms, n_transforms)) * 15000

    def _update_dirichlet_alphas(self, x_data, verbose):
        transform_batch_size = 512
        predict_batch_size = 1024
        n_transforms = self.transformer.n_transforms
        dirichlet_alphas = np.zeros((n_transforms, n_transforms))
        if verbose:
            print('Calculating dirichlet alphas...')
        for t_ind in tqdm(range(n_transforms), disable=not verbose):
            x_data_transformed, _ = self.transformer.apply_transforms(
                x_data, [t_ind], transform_batch_size)
            predictions_data = self.classifier.predict(
                x_data_transformed, predict_batch_size)
            dirichlet_alphas[t_ind, :] = dirichlet_utils.get_mle_alpha_t(
                predictions_data)
        self.dirichlet_alphas = dirichlet_alphas

    def update_prediction_threshold(self, x_validation, verbose=True):
        transform_batch_size = 512
        evaluation_batch_size = 1024
        print('Calculating prediction threshold...')
        dirichlet_scores_validation = self.predict_dirichlet_score(
            x_validation, transform_batch_size, evaluation_batch_size, verbose)
        self.prediction_threshold = np.percentile(
            dirichlet_scores_validation, 100 - self.percentile)

    def fit(self, x_train, epochs, x_validation=None, batch_size=128,
        iterations_to_validate=None, patience=None, verbose=True,
        iterations_to_print_train=None):
        print_manager = PrintManager().verbose_printing(verbose)
        self.transformer.verbose = verbose
        transform_batch_size=512
        if epochs is None:
            epochs = self._get_original_paper_epochs()
            patience = int(1e100)
        #transform datasets
        x_train_transform, y_train_transform = \
            self.transformer.apply_all_transforms(
                x=x_train, batch_size=transform_batch_size)
        validation_data = None
        if x_validation is not None:
            x_val_transform, y_val_transform = \
                self.transformer.apply_all_transforms(
                    x=x_validation, batch_size=transform_batch_size)
            validation_data = (
                x_val_transform, tf.keras.utils.to_categorical(y_val_transform))
        # fit classifier
        self.classifier.fit(
            x_train_transform, tf.keras.utils.to_categorical(y_train_transform),
            epochs, validation_data, batch_size, iterations_to_validate,
            patience, verbose,
            iterations_to_print_train=iterations_to_print_train)
        # update dirichlet params
        self._update_dirichlet_alphas(x_train, verbose)
        if x_validation is None:
            warnings.warn("No x_validation, x_train will be used to calculate "
                          "prediction threshold, update it with "
                          "model.update_prediction_threshold(x_validation)")
            x_validation = x_train
        self.update_prediction_threshold(x_validation, verbose)
        self.save_model(self.classifier.best_model_weights_path)
        print_manager.close()

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
        divided_scores, divided_labels = self. \
            _assign_inlier_outlier_sections_to_scores_and_labels(scores, labels)
        fpr, tpr, roc_thresholds = roc_curve(divided_labels, divided_scores)
        roc_auc = auc(fpr, tpr)
        accuracies = accuracies_by_threshold(labels, scores, roc_thresholds)
        # 100-percentile is necesary because normal data is at the right of
        # anormal
        thr = self.prediction_threshold
        acc_at_percentil = accuracy_at_thr(labels, scores, thr)
        recall_outliers_at_percentil = recall_at_thr(labels, scores, thr,
                                                     over_outliers=True)
        precision_outliers_at_percentil = precision_at_thr(labels, scores, thr,
                                                           over_outliers=True)
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
                        'acc_at_percentil': acc_at_percentil,
                        'rec_out_at_percentil': recall_outliers_at_percentil,
                        'prec_out_at_percentil': precision_outliers_at_percentil
                        }
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
        get_specific_metrics=False, verbose=True, log_file='evaluate.log'):
        print_manager = PrintManager().verbose_printing(verbose)
        file = open(os.path.join(self.model_results_path, log_file), 'a')
        print_manager.file_printing(file)
        print('\nEvaluating model...')
        print(dataset_name)
        print(x_eval.shape)
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
                'roc_auc', 'acc_at_percentil', 'pr_auc_norm',
                'rec_out_at_percentil', 'prec_out_at_percentil'])
        metrics = self._filter_metrics_to_return(
            metrics, get_specific_metrics,
            keys_to_keep=['roc_auc', 'acc_at_percentil', 'pr_auc_norm',
                          'rec_out_at_percentil', 'prec_out_at_percentil'])
        metrics_save_path = self.get_metrics_save_path(
            general_keys.DIRICHLET, dataset_name, class_name)
        self._save_metrics_on_paths(
            additional_score_save_path_list, metrics_save_path,
            metrics, save_metrics)
        print("\nEvaluation time: %s" % utils.timer(
            start_time, time.time()))
        print_manager.close()
        file.close()
        return metrics

    # is path_checkpoint, beause weights name might change
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

    # is path_checkpoint, beause weights name might change
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

    #####[DOWN] Methods streaming_geotransform/geotransformbase.py##

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
        utils.check_path(results_folder_path)
        return results_folder_path

    def _get_original_paper_epochs(self):
        return int(np.ceil(200 / self.transformer.n_transforms))

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

    def _print_final_metrics(
        self, metrics: dict, keys_to_keep: list):
        message = '\nEvaluation metrics:'
        for key_i in keys_to_keep:
            message += ' %s %.6f,' % (key_i, metrics[key_i])
        message = message[:-1]
        print(message)

    def _filter_metrics_to_return(
        self, metrics: dict, get_specific_metrics, keys_to_keep: list):
        if get_specific_metrics:
            filtered_metrics = {
                your_key: metrics[your_key] for your_key in keys_to_keep}
            return filtered_metrics
        return metrics

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

    # def save_weights(self, path):
    #     self.classifier.save_weights(path)
    #
    # def load_weights(self, path, by_name=False):
    #     self.classifier.load_weights(path, by_name=by_name)

if __name__ == '__main__':
    from modules.data_loaders.hits_outlier_loaderv2 import HiTSOutlierLoaderv2
    from modules.geometric_transform import transformations_tf
    from parameters import loader_keys

    EPOCHS = 1000
    ITERATIONS_TO_VALIDATE = None
    ITERATIONS_TO_PRINT_TRAIN = None
    PATIENCE = 0
    VERBOSE = True

    utils.set_soft_gpu_memory_growth()
    # data load
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '..', 'datasets', 'thesis_data', 'hits',
            'hits_small_4c_tuples.pkl'),
    }
    data_loader = HiTSOutlierLoaderv2(hits_params, 'small_hits')
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = data_loader.get_outlier_detection_datasets()
    # data transformation
    transformer = transformations_tf.TransTransformer()
    print('n transformations: ', transformer.n_transforms)
    # model
    clf = DeepHitsv2(transformer.n_transforms)
    # model = GeoTransformBasev2(
    #     classifier=clf, transformer=transformer,
    #     results_folder_name='test_geotranform_basev2')
    # model.fit(
    #     x_train, epochs=EPOCHS,#None,#
    #     x_validation=x_val,#None,#
    #     iterations_to_validate=ITERATIONS_TO_VALIDATE, patience=PATIENCE,
    #     verbose=VERBOSE, iterations_to_print_train=ITERATIONS_TO_PRINT_TRAIN)
    # # model.update_prediction_threshold(x_val)
    # model.evaluate(
    #     x_test, y_test, data_loader.name, 'real',
    #     save_metrics=True, save_histogram=True, get_specific_metrics=True,
    #     verbose=VERBOSE)

    # load model from chkpt
    model = GeoTransformBasev2(
        classifier=clf, transformer=transformer,
        results_folder_name='test_geotranform_basev2')
    model_folder = os.path.join(
        PROJECT_PATH,
        'results/test_geotranform_basev2/GeoTransform_Basev2_deep_hitsv2_20201028-001230/')
    model_weights_path = os.path.join(model_folder, 'checkpoints',
                                      'best_weights.ckpt')
    model.load_model(model_weights_path)
    model.set_model_results_path(model_folder)
    model.evaluate(
        x_test, y_test, data_loader.name, 'real',
        save_metrics=True, save_histogram=True, get_specific_metrics=True,
        verbose=VERBOSE)
