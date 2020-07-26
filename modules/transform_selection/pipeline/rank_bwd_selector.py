"""
Rank Backward Transformation selector:
Inspired by Transformation Selection Literature, In contrary to Forward pass,
here we start with al transformations and we try ELIMINATING them one by one.
This is meant to work only hor HITS or ZTF variants
Transformation selection
pipeline
"""

import os
import sys

import numpy as np

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.transform_selection.pipeline.abstract_selector import \
    AbstractTransformationSelector
from models.transformer_od_simple_net import TransformODSimpleModel
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from scripts.transformation_ranking.fwd_bwd_ranking. \
    transformation_ranking_backward import BackwardsTransformRanker
from modules.data_loaders.ztf_small_outlier_loader import \
    ZTFSmallOutlierLoader
from parameters import param_keys, loader_keys
from typing import Callable
from modules.transform_selection.pipeline.rank_fwd_selector import \
    RankingForwardTransformationSelector


class RankingBackwardTransformationSelector(
    RankingForwardTransformationSelector):
    def __init__(self, train_epochs=1000, n_trains=10,
        transformations_from_scratch=False,
        training_model_constructor: Callable[
            [], TransformODSimpleModel] = TransformODSimpleModel,
        name='C3-Bwd-Rank', verbose=False):
        super().__init__(
            verbose=verbose, name=name)
        self.transformations_from_scratch = transformations_from_scratch
        self.training_model_constructor = training_model_constructor
        self.fwd_ranker = BackwardsTransformRanker({
            param_keys.EPOCHS: train_epochs,
            param_keys.TRAIN_N_TIMES: n_trains
        }, verbose_training=verbose)


if __name__ == '__main__':
    VERBOSE = True
    N_TRAINS = 2
    TRAIN_EPOCHS = 1
    FROM_SCRATCH = False
    from parameters import loader_keys, general_keys
    from modules.geometric_transform.transformer_for_ranking import \
        RankingTransformer

    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 10000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],  # [2],#
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

    data_loader = ztf_loader  # hits_loader  #

    (x_train, y_train), (
        x_val, y_val), _ = data_loader.get_outlier_detection_datasets()
    x_samples = x_train  # [...,-1][...,None]

    transformer = RankingTransformer()
    trf_selector = RankingBackwardTransformationSelector(
        train_epochs=TRAIN_EPOCHS, n_trains=N_TRAINS,
        transformations_from_scratch=FROM_SCRATCH, verbose=VERBOSE)
    print('Init N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
    transformer = trf_selector.get_selected_transformer(
        transformer, x_train, data_loader)
    print('Final N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
