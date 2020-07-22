"""
Abstract Transformation selector to operate on a transformation selection
pipeline
"""

import abc

import os
import sys

import numpy as np

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.utils import check_path

# TODO: to avoid bad practice of different constructor signature, create params
# TODO:
#  Transformation selection loading and saving, better to be relegated to pipeline as a whole
class AbstractTransformationSelector(abc.ABC):
    def __init__(self, verbose=False, name=''):
        self.verbose = verbose
        self.name = name
        # self.results_folder_path = \
        #     self._create_selected_transformation_tuples_save_folder()

    # def _create_selected_transformation_tuples_save_folder(self):
    #     results_folder_path = os.path.join(PROJECT_PATH, 'results',
    #                              'transformation_selectors', self.name)
    #     check_path(results_folder_path)
    #     return results_folder_path

    @abc.abstractmethod
    def get_selection_score_array(self, transformer: AbstractTransformer,
        x_data: np.array, dataset_name: str):
        return

    def _get_selected_transformations_tuples(
        self, transformer: AbstractTransformer,
        binary_array_transformations_to_remove: np.array):
        transformation_tuples = list(transformer.transformation_tuples[
                                     :])
        n_transformations = transformer.n_transforms
        for trf_indx in range(n_transformations):
            if binary_array_transformations_to_remove[trf_indx] == 1:
                transformation_to_remove = transformation_tuples[trf_indx]
                transformation_tuples.remove(transformation_to_remove)
        transformation_tuples = tuple(transformation_tuples)
        return transformation_tuples

    @abc.abstractmethod
    def _get_binary_array_of_transformations_to_remove(self,
        score_array: np.array):
        return

    # def _save_selected_transformaiton_tuples(
    #     self, selected_transformation_tuples: tuple, dataset_name: str,
    #     transformer: AbstractTransformer):
    #     save_file_name = '%s_%s_%i' % (
    #         dataset_name, transformer.name, transformer.n_transforms)
    #     save_path = os.p


    def get_selected_transformater_from_data(self,
        transformer: AbstractTransformer, x_data: np.array, dataset_name=''):
        selection_score = self.get_selection_score_array(transformer, x_data,
                                                         dataset_name)
        binary_array_transformations_to_remove = \
            self._get_binary_array_of_transformations_to_remove(
                selection_score)
        selected_transformation_tuples = \
            self._get_selected_transformations_tuples(
                transformer, binary_array_transformations_to_remove)
        transformer.set_transformations_to_perform(
            selected_transformation_tuples)
        return transformer
