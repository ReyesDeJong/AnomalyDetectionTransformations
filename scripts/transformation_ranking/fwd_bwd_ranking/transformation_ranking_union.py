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
from scripts.transformation_ranking.fwd_bwd_ranking.\
    transformation_ranking_forward import ForwardsTransformRanker
import numpy as np

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class UnionTransformRanker(BackwardsTransformRanker):

    def __init__(self, input_params=dict, verbose_training=False):
        super().__init__(input_params, verbose_training)
        self.backward_ranker = BackwardsTransformRanker(input_params,
                                                    verbose_training)
        self.forward_ranker = ForwardsTransformRanker(input_params,
                                                 verbose_training)

    def _get_default_results_folder_name(self):
        return 'ranking_union'

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
        print('Ranking Transformations\nInliers data_loader: '
              '%s\nOutlier data_loader: %s' % (
            data_loader.name, outliers_data_loader.name))
        best_fwd_transforms = self.forward_ranker.get_best_transformations(
            data_loader, outliers_data_loader, transformer, ModelClass)
        print('Selected Trf Fwd %i %s' % (
            len(best_fwd_transforms), str(best_fwd_transforms)))
        best_bwd_transforms = self.backward_ranker.get_best_transformations(
            data_loader, outliers_data_loader, transformer, ModelClass)
        print('Selected Trf Bwd %i %s' % (
            len(best_bwd_transforms), str(best_bwd_transforms)))
        union_transformations = self.operate_both_lists(
            best_fwd_transforms, best_bwd_transforms)
        print('Best Trf %s %i %s' % (
            self._get_default_results_folder_name(),
            len(union_transformations), str(union_transformations)))
        self.get_ground_truth_metric_from_transformations(
            data_loader, ModelClass, transformer,
            union_transformations)
        print('Total time usage: %s' % utils.timer(start_time, time.time()))
        self.print_manager.close()
        file.close()
        return union_transformations

    def operate_both_lists(self, best_fwd_transforms, best_bwd_transforms):
        # union
        union_transformations = np.array(
            tuple(best_fwd_transforms)+tuple(best_bwd_transforms))
        union_transformations = np.unique(
            union_transformations, axis=0).tolist()
        return union_transformations



def main():
    # METRIC_TO_RANK_ON = 'roc_auc'
    N_TRAIN_RUNS = 10
    EPOCHS_TO_USE = 1000
    VERBOSE_TRAINING = False
    VERBOSE = True

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

    union_ranker = UnionTransformRanker({
        param_keys.EPOCHS: EPOCHS_TO_USE,
        param_keys.TRAIN_N_TIMES: N_TRAIN_RUNS
    }, verbose_training=VERBOSE_TRAINING)
    transformer = RankingTransformer()
    print(union_ranker.rank_transformations(
        hits_loader, ztf_loader, TransformODSimpleModel, transformer,
        verbose=VERBOSE))
    print(union_ranker.rank_transformations(
        hits_loader, hits_loader, TransformODSimpleModel, transformer,
        verbose=VERBOSE))
    print(union_ranker.rank_transformations(
        ztf_loader, hits_loader, TransformODSimpleModel, transformer,
        verbose=VERBOSE))
    print(union_ranker.rank_transformations(
        ztf_loader, ztf_loader, TransformODSimpleModel, transformer,
        verbose=VERBOSE))


if __name__ == "__main__":
    from scripts.transformation_ranking.fwd_bwd_ranking import transformation_ranking_intersection
    main()
    transformation_ranking_intersection.main()
