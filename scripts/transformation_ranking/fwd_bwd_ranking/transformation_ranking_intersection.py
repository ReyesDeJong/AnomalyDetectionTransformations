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
    transformation_ranking_union import UnionTransformRanker
import numpy as np

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class InterectionTransformRanker(UnionTransformRanker):

    def __init__(self, input_params=dict, verbose_training=False):
        super().__init__(input_params, verbose_training)

    def _get_default_results_folder_name(self):
        return 'ranking_intersection'

    def operate_both_lists(self, best_fwd_transforms, best_bwd_transforms):
        # intersection
        inter_transformations = list(set(best_fwd_transforms).intersection(
            best_bwd_transforms))
        return inter_transformations



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

    inter_ranker = InterectionTransformRanker({
        param_keys.EPOCHS: EPOCHS_TO_USE,
        param_keys.TRAIN_N_TIMES: N_TRAIN_RUNS
    }, verbose_training=VERBOSE_TRAINING)
    transformer = RankingTransformer()
    # print(inter_ranker.rank_transformations(
    #     hits_loader, ztf_loader, TransformODSimpleModel, transformer,
    #     verbose=VERBOSE))
    # print(inter_ranker.rank_transformations(
    #     hits_loader, hits_loader, TransformODSimpleModel, transformer,
    #     verbose=VERBOSE))
    # print(inter_ranker.rank_transformations(
    #     ztf_loader, hits_loader, TransformODSimpleModel, transformer,
    #     verbose=VERBOSE))
    # print(inter_ranker.rank_transformations(
    #     ztf_loader, ztf_loader, TransformODSimpleModel, transformer,
    #     verbose=VERBOSE))
    print(inter_ranker.get_best_transformations(
        ztf_loader, ztf_loader, transformer))

if __name__ == "__main__":
    main()
