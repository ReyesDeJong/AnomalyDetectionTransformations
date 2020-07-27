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
import pandas as pd
from parameters import loader_keys, general_keys
import itertools


class TransformSelectorRawLogFIDMatrix(object):

    def __init__(self, threshold_magnitud_order=1,
        verbose=False, show=False, from_scratch=False):
        self.form_scratch = from_scratch
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
            raw_fid_score_x_y = self._get_fid_from_two_data_sets(
                x_transformed_x_ind, x_transformed_y_ind)
            matrix_raw_fid[trf_ind_x, trf_ind_y] += raw_fid_score_x_y
        return self._post_process_fid_matrix(matrix_raw_fid)

    def get_useful_trfs_matrix(
        self, data_loader: HiTSOutlierLoader, x_data, transformer: AbstractTransformer,
        transform_batch_size=512):
        matrix_folder_path = self._create_matrix_path(data_loader, transformer)
        useful_trf_matrix_path = os.path.join(
            matrix_folder_path, 'fid_useful_trf_matrix.pkl')
        if os.path.exists(useful_trf_matrix_path) and not self.form_scratch:
            useful_trf_matrix = pd.read_pickle(useful_trf_matrix_path)
        else:
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
            self._plot_clusters(transformer, diff_with_self_log_fid_matrix,
                                useful_transforms_matrix)
            useful_trf_matrix = self._post_process_fid_matrix(useful_transforms_matrix*1.0)
            self._save_useful_trf_matrix(useful_trf_matrix, matrix_folder_path)
        return useful_trf_matrix

    def _post_process_fid_matrix(self, matrix_score):
        """fill diagonal with -1, and triangle bottom with reflex of
        up"""
        for i_x in range(matrix_score.shape[-2]):
            for i_y in range(matrix_score.shape[-1]):
                if i_x == i_y:
                    matrix_score[i_x, i_y] = -1
                elif i_x > i_y:
                    matrix_score[i_x, i_y] = matrix_score[i_y, i_x]
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

    def _plot_clusters(self, transformer, trfs_scores_matrix,  useful_transformations_matrix):
        if self.show:
            transformation_index_tuples = self._get_transformations_index_tuples(
                transformer)
            labels = []
            scores = []
            for x_y_tuple in transformation_index_tuples:
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

    def _create_matrix_path(self, data_loader: HiTSOutlierLoader,
        transformer: AbstractTransformer):
        matrix_folder_path = os.path.join(
            PROJECT_PATH, 'results', 'transformation_selection_fid',
            data_loader.name, '%s_%i' % (transformer.name,
                                         transformer.n_transforms)
        )
        utils.check_path(matrix_folder_path)
        return matrix_folder_path


    def get_transform_selection_transformer(
        self, data_loader: HiTSOutlierLoader, x_data,
        transformer: AbstractTransformer):
        useful_trf_matrix = self.get_useful_trfs_matrix(data_loader, x_data, transformer)
        transformer_selected = self._get_transformer_with_selected_transforms(
            useful_trf_matrix, transformer)
        return transformer_selected

    def _save_useful_trf_matrix(self, useful_trf_matrix, matrix_folder_path):
        utils.save_pickle(
            useful_trf_matrix,
            os.path.join(matrix_folder_path, 'fid_useful_trf_matrix.pkl'))
        # TODO: save plots
        utils.save_2d_image(
            useful_trf_matrix, 'gif_useful_trf_matrix',
            matrix_folder_path, axis_show='on')

    def _get_transformer_with_selected_transforms(self,
        useful_trfs_matrix: np.ndarray,
        transformer: AbstractTransformer):
        index_tuples = self._get_transformations_index_tuples(transformer)
        redundant_transforms_tuples = []
        for x_y_tuple in index_tuples:
            x_ind = x_y_tuple[0]
            y_ind = x_y_tuple[1]
            x_y_is_useful = useful_trfs_matrix[x_ind, y_ind]
            if not x_y_is_useful:
                redundant_transforms_tuples.append(x_y_tuple)
        if self.verbose:
            print('Conflicting transformations')
            for conflicting_tuple in redundant_transforms_tuples:
                print('(%i,%i): %s ; %s' % (
                    conflicting_tuple[0], conflicting_tuple[1],
                    str(transformer.transformation_tuples[
                            conflicting_tuple[0]]),
                    str(transformer.transformation_tuples[
                            conflicting_tuple[1]])))
        # TODO: do a random selection and, selection_accuracy_tolerance=accuracy_selection_tolerance) a most repeated based. THIS is first
        #  chosen
        transforms_to_delete = [x_y_tuple[1] for x_y_tuple in
                                redundant_transforms_tuples]
        unique_transforms_to_delete = np.unique(transforms_to_delete)
        if self.verbose:
            print('transforms_to_delete %s' % str(transforms_to_delete))
            print('unique transforms_to_delete %s' % str(unique_transforms_to_delete))
        reversed_unique_transfors_to_delete = unique_transforms_to_delete[
                                              ::-1]
        transformer.transformation_tuples = list(
            transformer.transformation_tuples)
        for i in reversed_unique_transfors_to_delete:
            del transformer.transformation_tuples[i]
            del transformer._transformation_ops[i]
        if self.verbose:
            print(
                'Left Transformations %i' % len(
                    transformer.transformation_tuples))
            print(transformer.transformation_tuples)
            print(len(transformer._transformation_ops))
        return transformer




if __name__ == "__main__":
    from modules.geometric_transform.transformations_tf import \
        PlusKernelTransformer, Transformer
    from modules.geometric_transform.transformer_for_ranking import \
        RankingTransformer
    from modules.data_loaders.ztf_small_outlier_loader import \
        ZTFSmallOutlierLoader
    from modules.geometric_transform.transformer_no_compositions import NoCompositionTransformer

    VERBOSE = False
    SHOW = False

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
        hits_loader_4c,
        ztf_loader_3c
    ]
    # transformer = RankingTransformer()
    transformer_no_comp = NoCompositionTransformer()
    transformer99 = PlusKernelTransformer()
    transformer72 = Transformer()
    transformers = [
        transformer_no_comp,
        # transformer72,
        # transformer99
    ]

    fid_selector = TransformSelectorRawLogFIDMatrix(
        verbose=VERBOSE, show=SHOW
    )
    import matplotlib
    matplotlib.use('Agg')
    for transformer_i in transformers:
        for data_loader_i in data_loaders:
            print('Original n transforms %i' % transformer_i.n_transforms)
            orig_trf = transformer_i.transformation_tuples[:]
            x_train = data_loader_i.get_outlier_detection_datasets()[0][0]
            print(x_train.shape)
            matrix = fid_selector.get_useful_trfs_matrix(data_loader_i,
                x_train, transformer_i)
            transformer_i = fid_selector.get_transform_selection_transformer(
                data_loader_i, x_train, transformer_i
            )
            # plt.imshow(matrix)
            # plt.colorbar()
            # plt.show()
            print('')
            print('Final n transforms %i' % transformer_i.n_transforms)
            transformer_i.set_transformations_to_perform(orig_trf)