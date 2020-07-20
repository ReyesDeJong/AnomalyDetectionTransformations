"""
Abstract Transformation selector to operate on a transformation selection
pipeline
"""

import abc

from modules.geometric_transform.transformations_tf import AbstractTransformer


class AbstractTransformationSelector(abc.ABC):
    def __init__(self, transforms_from_file=True):
        self.transforms_from_file=transforms_from_file

    @abc.abstractmethod
    def get_selected_transformations(self, transformer: AbstractTransformer):
        return
