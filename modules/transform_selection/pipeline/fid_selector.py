"""
FID Transformation selector: Inspired by Frechet Inceptin Distance
It estimates Wd of the original
data wrt to a transformation, and checks if there is a 1 order magnitud
difference between W(x,x) and W(x,t(x)) to consider t(x) redundant or not
. Redundant transformation Eg: Flip on Astro.

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
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from modules.transform_selection.fid_modules.\
    transform_selector_fid_different_data_for_x import \
    TransformSelectorFRawLogFIDOtherDatAsX
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys


class FIDTransformationSelector(AbstractTransformationSelector):
    def __init__(self, name='C2-B-FID', verbose=False):
        super().__init__(
            verbose=verbose, name=name)
        self.fid_selector = TransformSelectorFRawLogFIDOtherDatAsX()

    def _list_of_lists_to_tuple_of_tuple(self, list_of_lists):
        tuple_of_tuples = tuple(tuple(x) for x in list_of_lists)
        return tuple_of_tuples

    def _get_selected_transformations_tuples(
        self, transformer: AbstractTransformer, x_data: np.array,
        dataset_loader: HiTSOutlierLoader):
        orig_trfs = transformer.transformation_tuples[:]
        x_data = self._get_large_x_data(dataset_loader)
        selected_trfs_list_of_lists = self.fid_selector.\
            get_selected_transformations(
            x_data, transformer, verbose=self.verbose)
        selected_trfs_tuples = self._list_of_lists_to_tuple_of_tuple(
            selected_trfs_list_of_lists)
        transformer.set_transformations_to_perform(orig_trfs)
        return selected_trfs_tuples

    def _get_large_hits_data(self):
        hits_params = {
            loader_keys.DATA_PATH: os.path.join(
                PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
            loader_keys.N_SAMPLES_BY_CLASS: 100000,
            loader_keys.TEST_PERCENTAGE: 0.0,
            loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
            loader_keys.USED_CHANNELS: [0, 1, 2, 3],
            loader_keys.CROP_SIZE: 21,
            general_keys.RANDOM_SEED: 42,
            loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
        }
        hits_loader = HiTSOutlierLoader(hits_params, pickles_usage=False)
        x_train_fid = \
            hits_loader.get_outlier_detection_datasets()[0][0]
        return x_train_fid

    def _get_large_ztf_data(self):
        ztf_params = {
            loader_keys.DATA_PATH: os.path.join(
                PROJECT_PATH,
                '../datasets/ALeRCE_data/converted_pancho_septiembre.pkl'),
            loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
            loader_keys.USED_CHANNELS: [0, 1, 2],
            loader_keys.CROP_SIZE: 21,
            general_keys.RANDOM_SEED: 42,
            loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
        }
        ztf_loader = ZTFOutlierLoader(ztf_params, pickles_usage=False)
        x_train_fid = \
            ztf_loader.get_outlier_detection_datasets()[0][0]
        return x_train_fid

    def _get_large_x_data(self, dataset_loader: HiTSOutlierLoader):
        self.print_manager.verbose_printing(False)
        if 'hits' in dataset_loader.name:
            x_data = self._get_large_hits_data()
        elif 'ztf' in dataset_loader.name:
            x_data = self._get_large_ztf_data()
        else:
            x_data = None
        self.print_manager.verbose_printing(self.verbose)
        return x_data


if __name__ == '__main__':
    VERBOSE = True
    from modules.geometric_transform.transformer_for_ranking import \
        RankingTransformer
    from modules.data_loaders.ztf_small_outlier_loader import \
        ZTFSmallOutlierLoader

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
    trf_selector = FIDTransformationSelector(verbose=VERBOSE)
    print('Init N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
    transformer = trf_selector.get_selected_transformer(
        transformer, x_train, data_loader)
    print('Final N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
