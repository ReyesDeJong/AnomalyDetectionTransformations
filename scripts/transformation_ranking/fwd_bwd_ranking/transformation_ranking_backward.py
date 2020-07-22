"""
Training a model with basic-non-composed transforms, to visualize it feature
 layer and
then calculate FID
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform.transformer_for_ranking import \
    RankingTransformer
from models.transformer_od_simple_net import TransformODSimpleModel
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys, param_keys
import numpy as np
from modules import utils
import tensorflow as tf
from typing import Callable, List
from modules.trainer import ODTrainer
from modules.print_manager import PrintManager
import datetime
import time
import copy
from models.transformer_od import TransformODModel

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BackwardsTransformRanker(object):

    def __init__(self, input_params=dict, verbose_training=False):
        params = self._get_default_parameters()
        params.update(input_params)
        self.score_name = params[param_keys.SCORE_NAME]
        self.metric_name_to_retrieve = params[param_keys.METRIC_NAME]
        self.train_epochs = params[param_keys.EPOCHS]
        self.train_n_times = params[param_keys.TRAIN_N_TIMES]
        self.verbose_training = verbose_training
        self.print_manager = PrintManager()
        self.results_path = os.path.join(
            PROJECT_PATH, 'scripts', 'transformation_ranking',
            'fwd_bwd_ranking', 'aux_results',
            params[param_keys.RESULTS_FOLDER_NAME])
        utils.check_path(self.results_path)

    def _get_default_parameters(self):
        default_params = {
            param_keys.SCORE_NAME: general_keys.DIRICHLET,
            param_keys.METRIC_NAME: general_keys.ROC_AUC,
            param_keys.RESULTS_FOLDER_NAME:
                self._get_default_results_folder_name()
        }
        return default_params

    def _get_default_results_folder_name(self):
        return 'ranking_backward'

    def _get_training_data(self, data_loader: HiTSOutlierLoader):
        (x_train, y_train), (
            x_val, y_val), _ = data_loader.get_outlier_detection_datasets()
        return x_train, x_val

    def _get_test_data(self, data_loader: HiTSOutlierLoader,
        outliers_data_loader: HiTSOutlierLoader):
        _, _, (
            x_test, y_test) = data_loader.get_outlier_detection_datasets()
        ground_truth_outliers = x_test[y_test != 1]
        test_inliers = x_test[y_test == 1]
        # check if mor HiTS data is needed, as ZTF has 6k test samples and
        # HiTS only 4k test samples
        if ('ztf' in data_loader.name) and \
            ('hits' in outliers_data_loader.name):
            outliers_data_loader.n_samples_by_class = 10000
            outliers_data_loader.test_percentage_all_data = 0.3
            outliers_data_loader.set_pickles_usage(False)
        _, _, (x_test_other, y_test_other) = \
            outliers_data_loader.get_outlier_detection_datasets()
        other_set_outliers = x_test_other[y_test_other != 1]
        other_set_outliers_size_match = other_set_outliers[
                                        :len(ground_truth_outliers)]
        other_set_outliers_shape_match = self._match_sample_shapes(
            other_set_outliers_size_match, test_inliers)
        final_x_test = np.concatenate(
            [test_inliers, other_set_outliers_shape_match], axis=0)
        final_y_test = np.concatenate(
            [y_test[y_test == 1], y_test[y_test != 1]], axis=0)
        return final_x_test, final_y_test

    def _match_sample_shapes(self, unmatched_samples, final_shape_samples):
        # match height and width dims
        if unmatched_samples.shape[1] != final_shape_samples.shape[1] or \
            unmatched_samples.shape[2] != final_shape_samples.shape[2]:
            unmatched_samples = tf.image.resize(unmatched_samples,
                                                final_shape_samples.shape[
                                                1:3]).numpy()
        # match channels dims
        if unmatched_samples.shape[-1] <= final_shape_samples.shape[-1]:
            while unmatched_samples.shape[-1] != final_shape_samples.shape[-1]:
                unmatched_samples = np.concatenate(
                    [unmatched_samples, unmatched_samples[..., -1][..., None]],
                    axis=-1)
        else:
            unmatched_samples = unmatched_samples[...,
                                :final_shape_samples.shape[-1]]
        unmatched_samples = utils.normalize_by_channel_1_1(unmatched_samples)
        return unmatched_samples

    def _init_best_ran_metric(self,
        ModelClass: Callable[[], TransformODSimpleModel],
        transformer: RankingTransformer, x_train, x_val, x_test, y_test):
        model_trainer = ODTrainer({param_keys.EPOCHS: self.train_epochs})
        model_trainer.train_and_evaluate_model_n_times(
            ModelClass, transformer, x_train, x_val, x_test, y_test,
            self.train_n_times, self.verbose_training)
        result_mean, result_var = model_trainer.get_metric_mean_and_std()
        return result_mean

    def _get_log_file(self, data_loader: HiTSOutlierLoader,
        outliers_data_loader: HiTSOutlierLoader,
        transformer: RankingTransformer):
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        date = date.split('-')[0]
        log_file_name = '%s_%s_OE%s_%s%i_%s.txt' % (
            self._get_default_results_folder_name(), data_loader.name,
            outliers_data_loader.name, transformer.name,
            transformer.n_transforms, date
        )
        file = open(os.path.join(self.results_path, log_file_name), 'w')
        return file

    def rank_transformations(
        self, data_loader: HiTSOutlierLoader,
        outliers_data_loader: HiTSOutlierLoader,
        ModelClass: Callable[[], TransformODSimpleModel],
        transformer: RankingTransformer, verbose=True):
        transformer = copy.deepcopy(transformer)
        start_time = time.time()
        self.print_manager.verbose_printing(verbose)
        file = self._get_log_file(data_loader, outliers_data_loader,
                                  transformer)
        self.print_manager.file_printing(file)
        print('Ranking Transformations\nInliers data_loader:'
              ' %s\nOutlier data_loader: %s' % (
                  data_loader.name, outliers_data_loader.name))
        x_train, x_val = self._get_training_data(data_loader)
        x_test, y_test = self._get_test_data(data_loader, outliers_data_loader)
        best_rank_metric = self._init_best_ran_metric(
            ModelClass, transformer, x_train, x_val, x_test, y_test)
        original_transformation_list = transformer.transformation_tuples
        current_pool_of_transformations = original_transformation_list[:]
        best_transformations = original_transformation_list[:]
        print('\nInitial best transformations %i %s\n%.5f' % (
            len(best_transformations), str(best_transformations),
            best_rank_metric
        ))
        for leaving_out_step_i in range(len(original_transformation_list) - 2):
            time_usage = utils.timer(start_time, time.time())
            print('\n%s current transformation pool ' % time_usage,
                  len(current_pool_of_transformations),
                  current_pool_of_transformations)
            possible_transformations_to_leave_out = \
                current_pool_of_transformations[1:]
            best_rank_metric_in_this_leaving_out_round = 0
            for transformation_i_lo_leave_out in \
                possible_transformations_to_leave_out:
                trf_to_perform = list(current_pool_of_transformations[:])
                trf_to_perform.remove(transformation_i_lo_leave_out)
                trf_to_perform = tuple(trf_to_perform)
                # print(len(trf_to_perform), trf_to_perform)
                transformer.set_transformations_to_perform(trf_to_perform)
                model_trainer = ODTrainer(
                    {param_keys.EPOCHS: self.train_epochs})
                model_trainer.train_and_evaluate_model_n_times(
                    ModelClass, transformer, x_train, x_val, x_test, y_test,
                    self.train_n_times, self.verbose_training)
                result_mean, result_var = model_trainer.get_metric_mean_and_std()
                if result_mean > best_rank_metric:
                    best_rank_metric = result_mean
                    best_transformations = trf_to_perform
                    print('[BEST OF ALL] %i %s\n%.5f' % (
                        len(best_transformations), best_transformations,
                        best_rank_metric))
                if result_mean > best_rank_metric_in_this_leaving_out_round:
                    best_rank_metric_in_this_leaving_out_round = result_mean
                    best_trf_in_round = trf_to_perform
                    trf_removed_in_round = transformation_i_lo_leave_out
                    print('best in this round %i %s\n%.5f' % (
                        len(best_trf_in_round), best_trf_in_round,
                        best_rank_metric_in_this_leaving_out_round))
            print('Transformation removed in round %i %s' % (
                len(best_trf_in_round), trf_removed_in_round))
            current_pool_of_transformations = best_trf_in_round
        print('\nBest Trf    %s_%s %i %s\n%f' % (
            data_loader.name, outliers_data_loader.name,
            len(best_transformations), str(best_transformations),
            best_rank_metric))
        self._best_rank_metric = best_rank_metric
        self._best_transformations = best_transformations
        self.get_ground_truth_metric_from_transformations(
            data_loader, ModelClass, transformer,
            original_transformation_list)
        self.get_ground_truth_metric_from_transformations(
            data_loader, ModelClass, transformer,
            self._best_transformations)
        print('Total time usage: %s' % utils.timer(start_time, time.time()))
        self.print_manager.close()
        file.close()
        return self._best_transformations

    def _get_rank_already_performed_file_path(
        self, data_loader: HiTSOutlierLoader,
        outliers_data_loader: HiTSOutlierLoader,
        transformer: RankingTransformer):
        file_names = [f for f in os.listdir(self.results_path) if
                      os.path.isfile(os.path.join(self.results_path, f))]
        log_file_name_acronym = '%s_%s_OE%s_%s%i_' % (
            self._get_default_results_folder_name(), data_loader.name,
            outliers_data_loader.name, transformer.name,
            transformer.n_transforms
        )
        wanted_rank_file_name_list = [name for name in file_names if
                                      log_file_name_acronym in name]
        if len(wanted_rank_file_name_list) == 0:
            return None
        wanted_rank_file_name_list = np.sort(wanted_rank_file_name_list)
        return os.path.join(self.results_path, wanted_rank_file_name_list[0])

    def get_best_transformations(
        self, data_loader: HiTSOutlierLoader,
        outliers_data_loader: HiTSOutlierLoader,
        transformer: RankingTransformer, ModelClass=TransformODSimpleModel):
        rank_file_path = self._get_rank_already_performed_file_path(
            data_loader, outliers_data_loader, transformer)
        if rank_file_path is None or self._get_best_tranforms_from_file_line(
            rank_file_path) is None:
            self.rank_transformations(
                data_loader, outliers_data_loader, ModelClass,
                transformer, verbose=True)
            return self._best_transformations
        else:
            return self._get_best_tranforms_from_file_line(rank_file_path)

    def _get_best_tranforms_from_file_line(self, file_path):
        with open(file_path) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        line_with_transforms_list = [
            x.strip() for x in content if 'Best Trf' in x]
        if len(line_with_transforms_list) == 0:
            return None
        line_with_transforms = line_with_transforms_list[-1]
        begining_of_trf_pos = [
            pos for pos, char in enumerate(line_with_transforms) if
            char == '('][0]
        end_of_trf_pos = [
            pos for pos, char in enumerate(line_with_transforms) if
            char == ')'][-1]
        best_transform_as_txt = line_with_transforms[
                                begining_of_trf_pos:end_of_trf_pos + 1]
        return self._best_tranform_txt_to_tuple(best_transform_as_txt)

    def _best_tranform_txt_to_tuple(self, best_trf_txt):
        remove_begin_end = best_trf_txt[1:-1]
        tuples_non_formated_splitted = remove_begin_end.split(')')
        individual_transform_tuples = []
        for non_formated_tuple_i in tuples_non_formated_splitted:
            ints_list = [int(char_i) for char_i in non_formated_tuple_i if
                         self._string_is_digit(char_i)]
            individual_transform_tuples.append(tuple(ints_list))
        individual_transform_tuples = [tuple_i for tuple_i in
                                       individual_transform_tuples if
                                       len(tuple_i) != 0]
        return individual_transform_tuples

    def _string_is_digit(self, s: str):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def get_ground_truth_metric_from_transformations(
        self, data_loader: HiTSOutlierLoader,
        ModelClass: Callable[[], TransformODSimpleModel],
        transformer: RankingTransformer, transformation_list: List):
        transformer.set_transformations_to_perform(transformation_list)
        (x_train, y_train), (x_val, y_val), (
            x_test, y_test) = data_loader.get_outlier_detection_datasets()
        model_trainer = ODTrainer(
            {param_keys.EPOCHS: self.train_epochs})
        model_trainer.train_and_evaluate_model_n_times(
            TransformODModel, transformer, x_train, x_val, x_test, y_test,
            self.train_n_times, self.verbose_training)
        result_mean, result_var = model_trainer.get_metric_mean_and_std()
        print('Hardcoded WRN model usage')
        print('\nGround truth %i %s_%s %.5f+/-%.5f' % (
            len(transformation_list), data_loader.name, 'gt_outliers',
            result_mean, result_var))
        return result_mean, result_var


def main():
    # METRIC_TO_RANK_ON = 'roc_auc'
    N_TRAIN_RUNS = 10
    EPOCHS_TO_USE = 1000
    VERBOSE_TRAINING = False
    VERBOSE = False

    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 10000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    hits_loader = HiTSOutlierLoader(hits_params)
    ztf_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH,
            '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
    }
    ztf_loader = ZTFSmallOutlierLoader(ztf_params)

    bwd_ranker = BackwardsTransformRanker({
        param_keys.EPOCHS: EPOCHS_TO_USE,
        param_keys.TRAIN_N_TIMES: N_TRAIN_RUNS
    }, verbose_training=VERBOSE_TRAINING)
    transformer = RankingTransformer()
    print(bwd_ranker.rank_transformations(
        hits_loader, ztf_loader, TransformODSimpleModel, transformer,
        verbose=VERBOSE))
    print(bwd_ranker.rank_transformations(
        hits_loader, hits_loader, TransformODSimpleModel, transformer,
        verbose=VERBOSE))
    print(bwd_ranker.rank_transformations(
        ztf_loader, hits_loader, TransformODSimpleModel, transformer,
        verbose=VERBOSE))
    print(bwd_ranker.rank_transformations(
        ztf_loader, ztf_loader, TransformODSimpleModel, transformer,
        verbose=VERBOSE))
    print(bwd_ranker.get_best_transformations(
        ztf_loader, ztf_loader, transformer))


if __name__ == "__main__":
    main()