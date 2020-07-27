"""
Training a model with basic-non-composed transforms, to visualize it feature
 layer and
then calculate FID
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.transform_selection.fid_modules import fid
import numpy as np
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from modules import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from parameters import loader_keys, general_keys
import itertools


class TransformSelectorRawLogFIDMatrix(object):

    def __init__(self, threshold_magnitud_order=1, verbose=False, show=False):
        self.verbose = verbose
        self.show = show
        self._generator_name = 'RawLogFIDMatrix'
        self.threshold_magnitud_order = threshold_magnitud_order

    def _get_transformations_index_tuples(self, transformer: AbstractTransformer):
        transforms_arange = np.arange(transformer.n_transforms)
        transforms_tuples = list(
            itertools.product(transforms_arange, transforms_arange))
        transformation_index_tuples = []
        for x_y_tuple in transforms_tuples:
            if x_y_tuple[0] < x_y_tuple[1]:
                transformation_index_tuples.append(x_y_tuple)
        return transformation_index_tuples

    def _get_binary_data(self, transformer: AbstractTransformer, x_data, x_ind,
        y_ind, transform_batch_size):
        x_transformed_x_ind, _ = \
            transformer.apply_transforms(x_data, [x_ind], transform_batch_size)
        x_transformed_y_ind, _ = \
            transformer.apply_transforms(x_data, [y_ind], transform_batch_size)
        return x_transformed_x_ind, x_transformed_y_ind

    def get_raw_fid_matrix(
        self, x_data, transformer: AbstractTransformer,
        transform_batch_size=512):
        print('\nPredicting Acc Matrix\n')
        n_transforms = transformer.n_transforms
        transformation_index_tuples = self._get_transformations_index_tuples(
            transformer)
        matrix_raw_fid = np.zeros((n_transforms, n_transforms))
        for x_y_tuple in tqdm(transformation_index_tuples):
            trf_ind_x = x_y_tuple[0]
            trf_ind_y = x_y_tuple[1]
            x_transformed_x_ind, x_transformed_y_ind = self._get_binary_data(
                transformer, x_data, trf_ind_x, trf_ind_y, transform_batch_size)
            raw_fid_score_x_y = self._get_fid_moments_from_data(
                x_transformed_x_ind, x_transformed_y_ind)
            matrix_raw_fid[trf_ind_x, trf_ind_y] += raw_fid_score_x_y
        return self._post_process_fid_matrix(matrix_raw_fid)

    def get_useful_trfs_matrix(
        self, x_data, transformer: AbstractTransformer,
        transform_batch_size=512):
        raw_fid_matrix = self.get_raw_fid_matrix(
            x_data, transformer, transform_batch_size)
        n_samples_in_x = len(x_data)
        x_dataset_T0_1 = x_data[:n_samples_in_x//2]
        x_dataset_T0_2 = x_data[n_samples_in_x//2:]
        self_fid = self._get_fid_from_two_data_sets(x_dataset_T0_1,
                                                    x_dataset_T0_2)
        log_fid_matrix = np.log(raw_fid_matrix)
        diff_with_self_log_fid_matrix = np.abs(
            log_fid_matrix - np.log(self_fid))
        useful_transforms_matrix = diff_with_self_log_fid_matrix > \
                                  self.threshold_magnitud_order
        self._plot_clusters(diff_with_self_log_fid_matrix,
                            useful_transforms_matrix)
        return self._post_process_fid_matrix(useful_transforms_matrix)

    def _post_process_fid_matrix(self, matrix_score):
        """fill diagonal with -1, and triangle bottom with reflex of
        up"""
        for i_x in range(matrix_score.shape[-2]):
            for i_y in range(matrix_score.shape[-1]):
                if i_x == i_y:
                    matrix_score[:, i_x, i_y] = -1
                elif i_x > i_y:
                    matrix_score[:, i_x, i_y] = matrix_score[:, i_y, i_x]
        return matrix_score

    def _get_fid_from_two_data_sets(self, x_data_1, x_data_2):
        fid_moments_1 = self._get_fid_moments_from_data(x_data_1)
        fid_moments_2 = self._get_fid_moments_from_data(x_data_2)
        return self._get_fid_from_moments(fid_moments_1, fid_moments_2)

    def _get_fid_moments_from_data(self, data):
        mu, sigma = fid. \
            calculate_activation_statistics_from_activation_array(data)
        return (mu, sigma)

    def _get_fid_from_moments(self, fid_moments_1,
        fid_moments_2):
        fid_value = fid.calculate_frechet_distance(
            fid_moments_1[0], fid_moments_1[1],
            fid_moments_2[0],
            fid_moments_2[1])
        return fid_value

    def _plot_clusters(self, trfs_scores_matrix,  useful_transformations_matrix):
        if self.show:
            transformation_index_tuples = self._get_transformations_index_tuples(
                transformer)
            labels = []
            scores = []
            for x_y_tuple in tqdm(transformation_index_tuples):
                trf_ind_x = x_y_tuple[0]
                trf_ind_y = x_y_tuple[1]
                labels.append(useful_transformations_matrix[
                                           trf_ind_x, trf_ind_y])
                scores.append(trfs_scores_matrix[
                                  trf_ind_x, trf_ind_y])
            print('Transformation scores: ', scores)
            plt.plot([1] * len(scores), scores, 'o')
            plt.show()
            plt.scatter([1] * len(scores), scores, c=labels)
            plt.show()




if __name__ == "__main__":
    from modules.geometric_transform.transformations_tf import \
        PlusKernelTransformer
    from modules.geometric_transform.transformer_for_ranking import \
        RankingTransformer
    from modules.data_loaders.ztf_small_outlier_loader import \
        ZTFSmallOutlierLoader

    VERBOSE = True
    SHOW = True

    utils.init_gpu_soft_growth()
    # data loaders
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 100000,
        loader_keys.TEST_PERCENTAGE: 0.0,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    hits_loader_4c = HiTSOutlierLoader(hits_params, pickles_usage=False)
    ztf_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH,
            '../datasets/ALeRCE_data/converted_pancho_septiembre.pkl'),
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
        loader_keys.USED_CHANNELS: [0, 1, 2],
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    ztf_loader_3c = ZTFOutlierLoader(ztf_params, pickles_usage=False)


    data_loaders = [
        # hits_loader_4c,
        ztf_loader_3c
    ]
    transformer = RankingTransformer()
    print('Original n transforms %i' % transformer.n_transforms)

    fid_selector = TransformSelectorRawLogFIDMatrix(
        verbose=VERBOSE, show=SHOW
    )

    for data_loader_i in data_loaders:
        x_train = data_loader_i.get_outlier_detection_datasets()[0][0]
        print(x_train.shape)
        matrix = fid_selector.get_useful_trfs_matrix(
            x_train, transformer)
        print('')
    print('Original n transforms %i' % transformer.n_transforms)
