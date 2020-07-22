"""
Abstract Transformation selector to operate on a transformation selection
pipeline
"""

import abc

from modules.geometric_transform.transformations_tf import AbstractTransformer
import numpy as np

#TODO: to avoid bad practice of different constructor signature, create params
class AbstractTransformationSelector(abc.ABC):
    def __init__(self, transforms_from_file=True, verbose=False):
        self.transforms_from_file=transforms_from_file
        self.verbose = verbose

    @abc.abstractmethod
    def get_selection_score_array(self, transformer: AbstractTransformer, x_data: np.array):
        return

    @abc.abstractmethod
    def _get_selected_transformations_tuples(
        self, transformer: AbstractTransformer,
        binary_array_transformations_to_remove: np.array):
        return

    @abc.abstractmethod
    def _get_binary_array_of_transformations_to_remove(self,
        score_array: np.array):
        return


    def get_selected_transformater_from_data(self,
        transformer: AbstractTransformer, x_data: np.array, dataset_name=''):
        selection_score = self.get_selection_score_array(transformer, x_data)
        binary_array_transformations_to_remove = \
            self._get_binary_array_of_transformations_to_remove(
                selection_score)
        selected_transformation_tuples = \
            self._get_selected_transformations_tuples(
                transformer, binary_array_transformations_to_remove)
        transformer.set_transformations_to_perform(
            selected_transformation_tuples)
        return transformer