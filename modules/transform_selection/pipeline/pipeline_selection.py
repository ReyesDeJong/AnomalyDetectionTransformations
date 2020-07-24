"""
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
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.transform_selection.pipeline.abstract_selector import \
    AbstractTransformationSelector
from modules.utils import check_path, save_pickle
from typing import List
from modules.print_manager import PrintManager
import pandas as pd


# TODO: disentangle mess with other functions and methods, and matrix savers
class PipelineTransformationSelection(object):
    def __init__(
        self, verbose_pipeline=False, verbose_selectors=False,
        selection_pipeline: List[AbstractTransformationSelector] = None):
        self.verbose_pipeline = verbose_pipeline
        self.verbose_selectors = verbose_selectors
        self.pipeline_transformation_selectors = selection_pipeline
        self.set_selectors_verbose(verbose_selectors)
        self.results_folder_path = \
            self._create_selected_transformation_tuples_save_folder()
        self.print_manager = PrintManager()

    def _create_selected_transformation_tuples_save_folder(self):
        results_folder_path = os.path.join(
            PROJECT_PATH, 'results', 'transformation_selection', 'pipelines')
        check_path(results_folder_path)
        return results_folder_path

    def get_pipeline_name(
        self, transformer: AbstractTransformer, data_loader: HiTSOutlierLoader,
        n_transforms_orig: int):
        pipeline_name = 'pipeline_%s_%s%i' % (
            data_loader.name, transformer.name, n_transforms_orig)
        for selector in self.pipeline_transformation_selectors:
            pipeline_name += '_%s' % selector.name
        return pipeline_name

    def set_selectors_verbose(self, verbose):
        self.verbose_selectors = verbose
        for selector in self.pipeline_transformation_selectors:
            selector.verbose = verbose

    def set_pipeline_verbose(self, verbose):
        self.verbose_pipeline = verbose

    def append_to_pipeline(self, selector: AbstractTransformationSelector):
        self.pipeline_transformation_selectors.append(selector)
        return self

    def set_pipeline(self, pipeline: List[AbstractTransformationSelector]):
        self.pipeline_transformation_selectors = pipeline
        self.set_selectors_verbose(self.verbose_selectors)

    def _save_selected_transformations(
        self, step: int, transformer: AbstractTransformer,
        n_transforms_orig: int, dataset_loader: HiTSOutlierLoader):
        pipeline_name = self.get_pipeline_name(transformer, dataset_loader,
                                               n_transforms_orig)
        pipeline_save_file_name = '%s_step%i.pkl' % (pipeline_name, step)
        save_path = os.path.join(self.results_folder_path,
                                 pipeline_save_file_name)
        save_pickle(transformer.transformation_tuples, save_path)

    def _get_pipeline_final_save_path(
        self, transformer: AbstractTransformer, n_transforms_orig,
        dataset_loader: HiTSOutlierLoader):
        pipeline_name = self.get_pipeline_name(
            transformer, dataset_loader, n_transforms_orig)
        pipeline_save_file_name = '%s_step%i.pkl' % (
            pipeline_name, len(self.pipeline_transformation_selectors) - 1)
        final_save_path = os.path.join(self.results_folder_path,
                                       pipeline_save_file_name)
        return final_save_path

    def _check_selected_transforms_in_initials(
                self, selected_transforms, initial_transforms):
        for selected_transform_i in selected_transforms:
            if selected_transform_i not in initial_transforms:
                raise ValueError(
                    'Transformation %s not contontained' % (
                        str(selected_transform_i)) +
                    'in original pool of transformations %s' % (
                        str(initial_transforms)
                    ))


    def get_selected_transformer(self,
        transformer: AbstractTransformer, x_data: np.array,
        dataset_loader: HiTSOutlierLoader, load_from_checkpoint=True):
        n_transforms_orig = transformer.n_transforms
        final_save_path = self._get_pipeline_final_save_path(
            transformer, n_transforms_orig, dataset_loader)
        if os.path.exists(final_save_path) and load_from_checkpoint:
            selected_transformations = pd.read_pickle(final_save_path)
            transformer.set_transformations_to_perform(selected_transformations)
            print('n selected transforms %i' % (transformer.n_transforms))
            return transformer
        self.print_manager.verbose_printing(self.verbose_pipeline)
        for step_i, selector in enumerate(
            self.pipeline_transformation_selectors):
            print('Transform Selection %s' % selector.name)
            initial_transforms = transformer.transformation_tuples[:]
            transformer = selector.get_selected_transformer(
                transformer, x_data, dataset_loader)
            selected_transforms = transformer.transformation_tuples[:]
            print('Step %i transforms %s' % (step_i, selected_transforms))
            self._check_selected_transforms_in_initials(
                selected_transforms, initial_transforms)
            print('n selected transforms %i' % (transformer.n_transforms))
        self._save_selected_transformations(
            step_i, transformer, n_transforms_orig, dataset_loader)
        self.print_manager.close()
        return transformer


if __name__ == '__main__':
    VERBOSE_PIPELINE = True
    VERBOSE_SELECTORS = True
    from parameters import loader_keys, general_keys
    from modules.geometric_transform.transformer_for_ranking import \
        RankingTransformer
    from modules.data_loaders.ztf_small_outlier_loader import \
        ZTFSmallOutlierLoader
    from modules.transform_selection.pipeline.trivial_selector import \
        TrivialTransformationSelector
    from modules.transform_selection.pipeline.fid_selector import \
        FIDTransformationSelector
    from modules.transform_selection.pipeline.rank_fwd_selector import \
        RankingForwardTransformationSelector

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

    data_loader = hits_loader  #ztf_loader  #

    (x_train, y_train), (
        x_val, y_val), _ = data_loader.get_outlier_detection_datasets()
    x_samples = x_train  # [...,-1][...,None]

    transformer = RankingTransformer()
    trf_selector_pipeline = \
        PipelineTransformationSelection(
            verbose_pipeline=VERBOSE_PIPELINE,
            verbose_selectors=VERBOSE_SELECTORS,
            selection_pipeline=[
                TrivialTransformationSelector(),
                FIDTransformationSelector(),
                RankingForwardTransformationSelector()
            ]
        )
    print('Init N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
    transformer = trf_selector_pipeline.get_selected_transformer(
        transformer, x_train, data_loader)
    print('Final N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))