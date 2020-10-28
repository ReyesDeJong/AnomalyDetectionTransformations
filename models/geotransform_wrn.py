"""
non streaming geotransform wrn
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
from models.geotransform_basev2 import GeoTransformBasev2
from modules.networks.wide_residual_networkv2 import WideResnetv2


class GeoTransformWRN(GeoTransformBasev2):
    def __init__(self, transformer: AbstractTransformer,
        results_folder_name=None, name='GeoTransform_WRN'):
        super().__init__(WideResnetv2(transformer.n_transforms),
                         transformer, results_folder_name, name)
