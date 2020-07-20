"""
Trivial Transformation selector: Check if the information of the original
data has been completely erased. Eg: Random noise transformation, multipliying
data by a constant
to operate on a transformation selection
pipeline
"""

import abc

from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.transform_selection.pipeline.abstract_selector import AbstractTransformationSelector

class TrivialTransformationSelector(AbstractTransformationSelector):
    def __init__(self, transforms_from_file=True):
        self.transforms_from_file=transforms_from_file

    @abc.abstractmethod
    def get_selected_transformations(self, transformer: AbstractTransformer):
        return
