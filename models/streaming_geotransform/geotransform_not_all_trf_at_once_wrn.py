"""
First attempt (train_step_tf2 like) to build geometric trasnformer for outlier detection in tf2
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
from models.streaming_geotransform.\
    geotransform_base_not_all_transformations_at_once import \
    GeoTransformBaseNotAllTransformsAtOnce
from modules.networks.streaming_network. \
    streaming_transformations_wide_resnet import \
    StreamingTransformationsWideResnet


class GeoTransformNotAllAtOnceWRN(GeoTransformBaseNotAllTransformsAtOnce):

    def __init__(self, n_channels: int, transformer: AbstractTransformer,
        results_folder_name=None, name='GeoTransform_Not_All_At_Once_WRN'):
        classifier = StreamingTransformationsWideResnet(
            n_channels, transformer)
        super().__init__(classifier, transformer, results_folder_name, name)
