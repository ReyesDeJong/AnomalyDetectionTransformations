"""
Trivial Transformation selector: Check if the information of the original
data has been completely erased. Eg: Random noise transformation, multipliying
data by a constant
to operate on a transformation selection
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
from modules.info_metrics.information_estimator_by_batch import \
    InformationEstimatorByBatch
from tqdm import tqdm


class TrivialTransformationSelector(AbstractTransformationSelector):
    def __init__(self, random_seed=42, n_samples_batch=512, sigma_zero=2.0,
        as_image=True, name='C1_MI',
        verbose=False):
        super().__init__(
            verbose=verbose, name=name)
        self.mi_estimator = InformationEstimatorByBatch(
            sigma_zero, n_samples_batch, random_seed,
            x_and_y_as_images=as_image)

    def get_MI_array(self, transformer: AbstractTransformer,
        x_data: np.array):
        n_transforms = transformer.n_transforms
        # print('N Trfs to analize: %i\n%s' %
        #      (n_transforms, str(transformer.transformation_tuples)))
        trfs_idexes = list(range(n_transforms))
        mean_mi_list = []
        for transformation_i in tqdm(trfs_idexes, disable=not self.verbose):
            # start_time = time.time()
            # print('Current processed tranformation %s' %
            #       str(transformer.transformation_tuples[transformation_i]))
            x_transformed, y_transformed = transformer.apply_transforms(
                x_data, [transformation_i])
            mean_mi_i = self.mi_estimator.mutual_information_mean_fast(
                x_transformed, x_data)
            mean_mi_list.append(mean_mi_i)
            # print('%s MI: %f' % (
            #     str(transformer.transformation_tuples[transformation_i]),
            #     mean_mi_i))
            # print(timer(start_time, time.time()))
        return np.array(mean_mi_list)

    def _get_binary_array_of_transformations_to_remove(self,
        mi_array: np.array):
        return np.abs(mi_array) < 0.001

    def get_selection_score_array(self, transformer: AbstractTransformer,
        x_data: np.array, dataset_loader: str):
        return self.get_MI_array(transformer, x_data)

    def get_selected_transformer(self,
        transformer: AbstractTransformer, x_data: np.array, dataset_loader=None):
        return super().get_selected_transformer(
            transformer, x_data, dataset_loader)


if __name__ == '__main__':
    VERBOSE = True
    from parameters import loader_keys, general_keys
    from modules.geometric_transform.transformer_for_ranking import \
        RankingTransformer
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
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
    trf_selector = TrivialTransformationSelector(verbose=VERBOSE)
    print('Init N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
    transformer = trf_selector.get_selected_transformer(transformer,
                                                        x_train)
    print('Final N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))
