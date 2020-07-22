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
from modules.info_metrics.information_estimator_v2 import InformationEstimator
from modules.info_metrics.information_estimator_by_batch import \
    InformationEstimatorByBatch
from tqdm import tqdm
import tensorflow as tf
import time


class TrivialTransformationSelector(AbstractTransformationSelector):
    def __init__(self, random_seed=42, n_samples_batch=512, sigma_zero=2.0,
        as_image=True, get_transforms_from_file_if_posible=True, verbose=False):
        super().__init__(
            transforms_from_file=get_transforms_from_file_if_posible,
            verbose=verbose)
        self.random_seed = random_seed
        self.n_samples_batch = n_samples_batch
        self.as_images = as_image
        self.estimator = InformationEstimator(sigma_zero)

    def get_MI_array(self, transformer: AbstractTransformer,
        x_data: np.array):
        n_transforms = transformer.n_transforms
        list('N Trfs to analize: %i\n%s' %
             (n_transforms, str(transformer.transformation_tuples)))
        trfs_idexes = list(range(n_transforms))
        mean_mi_list = []
        for transformation_i in tqdm(trfs_idexes, disable=not self.verbose):
            # print('Current processed tranformation %s' %
            #       str(transformer.transformation_tuples[transformation_i]))
            x_transformed, y_transformed = transformer.apply_transforms(
                x_data, [transformation_i])
            estimation_ds = tf.data.Dataset.from_tensor_slices(
                (x_data, x_transformed)).shuffle(
                10000, seed=self.random_seed).batch(
                self.n_samples_batch)
            batch_mi_list = []
            start_time = time.time()
            for x_orig, x_trans in estimation_ds:
                mi_estimation = self.estimator.mutual_information(
                    x_orig, x_trans, x_is_image=self.as_images,
                    y_is_image=self.as_images)
                batch_mi_list.append(mi_estimation.numpy())
            mean_mi_list.append(np.mean(batch_mi_list))
            # print('%s MI: %f+/-%f' % (
            #     str(transformer.transformation_tuples[transformation_i]),
            #     np.mean(batch_mi_list),
            #     np.std(batch_mi_list)))
            # print(timer(start_time, time.time()))
        return np.array(mean_mi_list)

    def _get_binary_array_of_transformations_to_remove(self,
        mi_array: np.array):
        return np.abs(mi_array) < 0.001

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

    def get_selection_score_array(self, transformer: AbstractTransformer, x_data: np.array):
        return self.get_MI_array(transformer, x_data)




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
    transformer = trf_selector.get_selected_transformater_from_data(transformer, x_train)
    print('Final N transforms %i\n%s' % (
        transformer.n_transforms, str(transformer.transformation_tuples)))