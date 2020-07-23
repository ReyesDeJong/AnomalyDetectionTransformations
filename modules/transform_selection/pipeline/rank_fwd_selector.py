"""
Rank Forward Transformation selector:
Inspired by Transformation Selection Literature, it test transformations one by
 one in an iterative manner, adding transformation to n initial pool of the
 original data. To keep a transformation it must improve AUROC based on a
 dataset of other type of outlier, not real outliers

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
from models.transformer_od import TransformODModel
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from scripts.transformation_ranking.fwd_bwd_ranking. \
    transformation_ranking_forward import ForwardsTransformRanker
from modules.data_loaders.ztf_small_outlier_loader import \
    ZTFSmallOutlierLoader
from parameters import param_keys, loader_keys
from typing import Callable


# TODO: disentangle mess with other functions and methods, and matrix savers
class RankingForwardTransformationSelector(AbstractTransformationSelector):
    def __init__(self, train_epochs=1000, n_trains=10,
        transformations_from_scratch=False,
        training_model_constructor: Callable[
            [], TransformODSimpleModel] = TransformODModel,
        name='C3-Fwd-Rank', verbose=False):
        super().__init__(
            verbose=verbose, name=name)
        self.transformations_from_scratch = transformations_from_scratch
        self.training_model_constructor = training_model_constructor
        self.fwd_ranker = ForwardsTransformRanker({
            param_keys.EPOCHS: train_epochs,
            param_keys.TRAIN_N_TIMES: n_trains
        }, verbose_training=verbose)

    def _list_of_lists_to_tuple_of_tuple(self, list_of_lists):
        tuple_of_tuples = tuple(tuple(x) for x in list_of_lists)
        return tuple_of_tuples

    def get_binary_array_of_rejected_transformations_by_ranking(
        self, dataset_loader: HiTSOutlierLoader,
        other_dataset_loader: HiTSOutlierLoader,
        transformer: AbstractTransformer):
        orig_trfs = transformer.transformation_tuples[:]
        if self.transformations_from_scratch:
            selected_trf_list = self.fwd_ranker.rank_transformations(
                dataset_loader, other_dataset_loader,
                self.training_model_constructor, transformer, self.verbose)
        else:
            selected_trf_list = self.fwd_ranker.get_best_transformations(
                dataset_loader, other_dataset_loader, transformer,
                self.training_model_constructor)
        selected_trfs = tuple(selected_trf_list)
        n_orig_transforms = len(orig_trfs)
        redundant_transforms = np.ones(n_orig_transforms)
        for trf_idx in range(len(orig_trfs)):
            # check if not redundant (not redundant transforms are 0)
            if orig_trfs[trf_idx] in selected_trfs:
                redundant_transforms[trf_idx] = 0
        return redundant_transforms

    def _get_binary_array_of_transformations_to_remove(self,
        rank_rejected_transformation_array: np.array):
        return rank_rejected_transformation_array

    def _get_hits_dataset(self) -> HiTSOutlierLoader:
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
        hits_loader = HiTSOutlierLoader(hits_params, pickles_usage=False)
        return hits_loader

    def _get_ztf_dataset(self) -> ZTFSmallOutlierLoader:
        ztf_params = {
            loader_keys.DATA_PATH: os.path.join(
                PROJECT_PATH,
                '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
        }
        ztf_loader = ZTFSmallOutlierLoader(ztf_params, pickles_usage=False)
        return ztf_loader

    def get_selection_score_array(self, transformer: AbstractTransformer,
        x_data: np.array, dataset_loader: HiTSOutlierLoader):
        self.print_manager.verbose_printing(False)
        if 'hits' in dataset_loader.name:
            other_dataset_loader = self._get_ztf_dataset()
        elif 'ztf' in dataset_loader.name:
            other_dataset_loader = self._get_hits_dataset()
        #TODO: fix this condition, to work with more than hits or ztf
        else:
            other_dataset_loader = None
        self.print_manager.verbose_printing(self.verbose)
        return self.get_binary_array_of_rejected_transformations_by_ranking(
            dataset_loader, other_dataset_loader, transformer)


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
    trf_selector = RankingForwardTransformationSelector(
        train_epochs=TRAIN_EPOCHS, n_trains=N_TRAINS,
        transformations_from_scratch=FROM_SCRATCH, verbose=VERBOSE)
    print('Init N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
    transformer = trf_selector.get_selected_transformer(
        transformer, x_train, data_loader)
    print('Final N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
