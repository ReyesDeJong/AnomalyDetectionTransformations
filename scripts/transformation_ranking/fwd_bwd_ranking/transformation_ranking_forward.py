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
from modules import utils
from typing import Callable
from modules.trainer import ODTrainer
import time
import copy
from scripts.transformation_ranking.fwd_bwd_ranking. \
    transformation_ranking_backward import BackwardsTransformRanker

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ForwardsTransformRanker(BackwardsTransformRanker):

    def __init__(self, input_params=dict, verbose_training=False):
        super().__init__(input_params, verbose_training)

    def _get_default_results_folder_name(self):
        return 'ranking_forward'

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
        print('Inliers data_loader: %s\nOutlier data_loader: %s' % (
            data_loader.name, outliers_data_loader.name))
        x_train, x_val = self._get_training_data(data_loader)
        x_test, y_test = self._get_test_data(data_loader, outliers_data_loader)
        best_rank_metric = 0
        original_transformation_list = transformer.transformation_tuples
        transformation_t0 = [original_transformation_list[0]]
        # next is current_transformations_pool in backwards
        transformations_to_rank = original_transformation_list[1:]
        for n_transformation_to_rank_i in range(len(transformations_to_rank)):
            time_usage = utils.timer(start_time, time.time())
            print('\n%s current transformation to rank pool ' % time_usage,
                  len(transformations_to_rank),
                  transformations_to_rank)
            best_rank_metric_in_this_step = 0
            for transformation_i_to_evaluate in transformations_to_rank:
                transformations_to_perform = transformation_t0 + [
                    transformation_i_to_evaluate]
                print(transformations_to_perform)
                transformer.set_transformations_to_perform(
                    transformations_to_perform)
                model_trainer = ODTrainer(
                    {param_keys.EPOCHS: self.train_epochs})
                model_trainer.train_and_evaluate_model_n_times(
                    ModelClass, transformer, x_train, x_val, x_test, y_test,
                    self.train_n_times, self.verbose_training)
                result_mean, result_var = model_trainer.get_metric_mean_and_std()
                if result_mean > best_rank_metric:
                    best_rank_metric = result_mean
                    best_transformations = transformations_to_perform
                    print('[BEST OF ALL] %i %s\n%.5f' % (
                        len(best_transformations), best_transformations,
                        best_rank_metric))
                if result_mean > best_rank_metric_in_this_step:
                    best_rank_metric_in_this_step = result_mean
                    best_trf_in_round = transformations_to_perform
                    trf_added_in_round = transformation_i_to_evaluate
                    print('best in this round %i %s\n%.5f' % (
                        len(best_trf_in_round), best_trf_in_round,
                        best_rank_metric_in_this_step))
            print('Transformation added in round %i %s' % (
                len(best_trf_in_round), trf_added_in_round))
            transformation_t0 = best_trf_in_round
            print(transformation_t0)
            transformations_to_rank = [x for x in transformations_to_rank if
                                       x not in transformation_t0]
            print(transformations_to_rank)
        print('\nBest Trf %s_%s %i %s\n%f' % (
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
        return self._best_transformations, self._best_rank_metric


def main():
    # METRIC_TO_RANK_ON = 'roc_auc'
    N_TRAIN_RUNS = 5
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

    bwd_ranker = ForwardsTransformRanker({
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


if __name__ == "__main__":
    main()
