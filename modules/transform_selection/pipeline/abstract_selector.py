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

from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.print_manager import PrintManager


# TODO: to avoid bad practice of different constructor signature, create params

# TODO: Transformation selection loading and saving,
#  better to be relegated to pipeline as a whole

# TODO: instead of order index base transformation selection, make tuple with
#  transformation to modify, it is more robust, otherwise indexes may get
#  through blind to the transformations available and erase unwantedthings,
#  it's no robust

# TODO: Name should be a property, not a contructor's input
class AbstractTransformationSelector(abc.ABC):
    def __init__(self, verbose=False, name=''):
        self.print_manager = PrintManager()
        self.verbose = verbose
        self.name = name

    @abc.abstractmethod
    def _get_selected_transformations_tuples(
        self, transformer: AbstractTransformer, x_data: np.array,
        dataset_loader: HiTSOutlierLoader):
        return

    def get_selected_transformer(self,
        transformer: AbstractTransformer, x_data: np.array,
        dataset_loader: HiTSOutlierLoader):
        self.print_manager.verbose_printing(self.verbose)
        selected_transformation_tuples = \
            self._get_selected_transformations_tuples(
                transformer, x_data, dataset_loader)
        transformer.set_transformations_to_perform(
            selected_transformation_tuples)
        self.print_manager.close()
        return transformer
