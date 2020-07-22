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