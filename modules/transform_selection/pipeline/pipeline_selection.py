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
from modules.transform_selection.pipeline.abstract_selector import \
    AbstractTransformationSelector
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from models.transformer_ensemble_ovo_simple_net_od import \
    EnsembleOVOTransformODSimpleModel
from scripts.transform_selection_disc_mat.training_transform_selection import \
    get_transform_selection_transformer
from modules.transform_selection.pipeline.identity_selector import \
    IdentityTransformationSelector
from modules.transform_selection.pipeline.abstract_selector import \
    AbstractTransformationSelector
from modules.utils import check_path
from typing import List


# TODO: disentangle mess with other functions and methods, and matrix savers
class PipelineTransformationSelection(object):
    def __init__(
        self, pipeline_verbose=False, selectors_verbose=False,
        selection_pipeline: List[AbstractTransformationSelector] = None):
        self.pipeline_verbose = pipeline_verbose
        self.selectors_verbose = selectors_verbose
        self.pipeline_transformation_selectors = selection_pipeline
        self.set_selectors_verbose(selectors_verbose)
        self.results_folder_path = \
            self._create_selected_transformation_tuples_save_folder()

    def _create_selected_transformation_tuples_save_folder(self):
        results_folder_path = os.path.join(
            PROJECT_PATH, 'results', 'transformation_selection', 'pipelines')
        check_path(results_folder_path)
        return results_folder_path

    def get_pipeline_name(
        self, transformer: AbstractTransformer, data_loader: HiTSOutlierLoader):
        pipeline_name = 'pipeline_%_%%i' % (
            data_loader.name, transformer.name, transformer.n_transforms)
        for selector in self.pipeline_transformation_selectors:
            pipeline_name += '_%s' % selector.name
        return pipeline_name

    def set_selectors_verbose(self, verbose):
        self.selectors_verbose = verbose
        for selector in self.pipeline_transformation_selectors:
            selector.verbose = verbose

    def set_pipeline_verbose(self, verbose):
        self.pipeline_verbose = verbose

    def append_to_pipeline(self, selector: AbstractTransformationSelector):
        self.pipeline_transformation_selectors.append(selector)
        return self

    def set_pipeline(self, pipeline: List[AbstractTransformationSelector]):
        self.pipeline_transformation_selectors = pipeline
        self.set_selectors_verbose(self.selectors_verbose)

    def _save_selected_transformations(
        self, step: int, transformer: AbstractTransformer,
        dataset_loader: HiTSOutlierLoader):
        pipeline_name = self.get_pipeline_name(transformer, dataset_loader)


    def get_selected_transformer(self,
        transformer: AbstractTransformer, x_data: np.array,
        dataset_loader: HiTSOutlierLoader):
        for step_i, selector in enumerate(
            self.pipeline_transformation_selectors):
            transformer = selector.get_selected_transformer(
                transformer, x_data, dataset_loader)
            self._save_selected_transformations(step_i, transformer,
                                                dataset_loader)
        return transformer


if __name__ == '__main__':
    VERBOSE = False
    from parameters import loader_keys, general_keys
    from modules.geometric_transform.transformer_for_ranking import \
        RankingTransformer
    from modules.data_loaders.ztf_small_outlier_loader import \
        ZTFSmallOutlierLoader
    from modules.transform_selection.pipeline.trivial_selector import \
        TrivialTransformationSelector
    from modules.transform_selection.pipeline.fid_selector import \
        FIDTransformationSelector

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
    trf_selector_pipeline = \
        PipelineTransformationSelection(
            pipeline_verbose=VERBOSE,
            selection_pipeline=[
                TrivialTransformationSelector(),
                FIDTransformationSelector()
            ]
        )
    print('Init N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
    transformer = trf_selector_pipeline.get_selected_transformer(
        transformer, x_train, data_loader)
    print('Final N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
