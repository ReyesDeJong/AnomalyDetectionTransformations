"""
Discrimination Matrix Transformation selector:
It compare all possible pairs of transformations and check if they are
discriminable by a Deep-HiTS-like classifier, if the classifier gets a ~50%
accuracy, then the transformations are not discriminable, then, deemed redundant
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

# TODO: disentangle mess with other functions and methods, and matrix savers
class DiscriminationMatrixTransformationSelector(AbstractTransformationSelector):
    def __init__(self, name='C2-A-DM', verbose=False):
        super().__init__(
            verbose=verbose, name=name)

    def _get_selected_transformations_tuples(
        self, transformer: AbstractTransformer, x_data: np.array,
        dataset_loader: HiTSOutlierLoader):
        orig_trfs = transformer.transformation_tuples[:]
        mdl = EnsembleOVOTransformODSimpleModel(
            data_loader=dataset_loader, transformer=transformer,
            input_shape=x_data.shape, verbose=self.verbose)
        transformer = get_transform_selection_transformer(data_loader, mdl,
                                                          transformer)
        selected_trfs_tuples = tuple(transformer.transformation_tuples)
        transformer.set_transformations_to_perform(orig_trfs)
        return selected_trfs_tuples


if __name__ == '__main__':
    VERBOSE = False
    from parameters import loader_keys, general_keys
    from modules.geometric_transform.transformer_for_ranking import \
        RankingTransformer
    from modules.data_loaders.ztf_small_outlier_loader import \
        ZTFSmallOutlierLoader

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
    trf_selector = DiscriminationMatrixTransformationSelector(verbose=VERBOSE)
    print('Init N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
    transformer = trf_selector.get_selected_transformer(
        transformer, x_train, data_loader)
    print('Final N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
