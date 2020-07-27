# """
# Training a model with basic-non-composed transforms, to visualize it feature
#  layer and
# then calculate FID
# """
#
# import os
# import sys
#
# PROJECT_PATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# sys.path.append(PROJECT_PATH)
#
# from modules.geometric_transform.transformations_tf import AbstractTransformer
# from modules.transform_selection.fid_modules import fid
# import numpy as np
# from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
# from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
# from modules import utils
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import pandas as pd
# from parameters import loader_keys, general_keys
# import itertools
# from modules.transform_selection.fid_modules.transform_selector_fid_matrix import TransformSelectorRawLogFIDMatrix
# import multiprocessing
# from joblib import Parallel, delayed
#
# #TODO: not working it would need to preload all transformations, bt dataset its too large
#
# def _get_raw_fid_from_x_y_tuple(
#     transformer: AbstractTransformer, x_y_tuple, x_data, transform_batch_size):
#     trf_ind_x = x_y_tuple[0]
#     trf_ind_y = x_y_tuple[1]
#     x_transformed_x_ind, x_transformed_y_ind = _get_binary_data(
#         transformer, x_data, trf_ind_x, trf_ind_y, transform_batch_size)
#     raw_fid_score_x_y = _get_fid_from_two_data_sets(
#         x_transformed_x_ind, x_transformed_y_ind)
#     return raw_fid_score_x_y
#
# def _get_binary_data(transformer: AbstractTransformer, x_data, x_ind,
#     y_ind, transform_batch_size):
#     x_transformed_x_ind, _ = \
#         transformer.apply_transforms(x_data, [x_ind], transform_batch_size)
#     x_transformed_y_ind, _ = \
#         transformer.apply_transforms(x_data, [y_ind], transform_batch_size)
#     return x_transformed_x_ind, x_transformed_y_ind
#
# def _get_fid_from_two_data_sets(x_data_1, x_data_2):
#     fid_moments_1 = _get_fid_moments_from_data(x_data_1)
#     fid_moments_2 = _get_fid_moments_from_data(x_data_2)
#     return _get_fid_from_moments(fid_moments_1, fid_moments_2)
#
# def _get_fid_moments_from_data(data):
#     mu, sigma = fid. \
#         calculate_activation_statistics_from_activation_array(data)
#     return (mu, sigma)
#
# def _get_fid_from_moments(fid_moments_1,
#     fid_moments_2):
#     fid_value = fid.calculate_frechet_distance(
#         fid_moments_1[0], fid_moments_1[1],
#         fid_moments_2[0],
#         fid_moments_2[1])
#     return fid_value
#
# class TransformSelectorRawLogFIDMatrixParallel(TransformSelectorRawLogFIDMatrix):
#
#     def __init__(self, threshold_magnitud_order=1,
#         verbose=False, show=False, from_scratch=False):
#         super().__init__(threshold_magnitud_order, verbose, show, from_scratch)
#         self.n_cpus = multiprocessing.cpu_count()
#
#     def get_raw_fid_matrix(
#         self, x_data, transformer: AbstractTransformer,
#         transform_batch_size=512):
#         print('\nPredicting Acc Matrix\n')
#         n_transforms = transformer.n_transforms
#         transformation_index_tuples = self._get_transformations_index_tuples(
#             transformer)
#         matrix_raw_fid = np.zeros((n_transforms, n_transforms))
#
#         raw_fid_scores_list = Parallel(n_jobs=self.n_cpus)(
#             delayed(_get_raw_fid_from_x_y_tuple)(
#                 transformer, transformation_index_tuples[i], x_data,
#                 transform_batch_size)
#             for
#             i in tqdm(range(len(transformation_index_tuples)),
#                       disable=not self.verbose))
#
#         for i, x_y_tuple in enumerate(transformation_index_tuples):
#             trf_ind_x = x_y_tuple[0]
#             trf_ind_y = x_y_tuple[1]
#             matrix_raw_fid[trf_ind_x, trf_ind_y] += raw_fid_scores_list[i]
#         return self._post_process_fid_matrix(matrix_raw_fid)
#
# if __name__ == "__main__":
#     from modules.geometric_transform.transformations_tf import \
#         PlusKernelTransformer
#     from modules.geometric_transform.transformer_for_ranking import \
#         RankingTransformer
#     from modules.data_loaders.ztf_small_outlier_loader import \
#         ZTFSmallOutlierLoader
#     from modules.geometric_transform.transformer_no_compositions import NoCompositionTransformer
#
#     VERBOSE = True
#     SHOW = True
#     FROM_SCRATCH = True
#
#     utils.init_gpu_soft_growth()
#     # data loaders
#     hits_params = {
#         loader_keys.DATA_PATH: os.path.join(
#             PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
#         loader_keys.N_SAMPLES_BY_CLASS: 100000,
#         loader_keys.TEST_PERCENTAGE: 0.0,
#         loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
#         loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
#         loader_keys.CROP_SIZE: 21,
#         general_keys.RANDOM_SEED: 42,
#         loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
#     }
#     hits_loader_4c = HiTSOutlierLoader(hits_params, pickles_usage=False)
#     ztf_params = {
#         loader_keys.DATA_PATH: os.path.join(
#             PROJECT_PATH,
#             '../datasets/ALeRCE_data/converted_pancho_septiembre.pkl'),
#         loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
#         loader_keys.USED_CHANNELS: [0, 1, 2],
#         loader_keys.CROP_SIZE: 21,
#         general_keys.RANDOM_SEED: 42,
#         loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
#     }
#     ztf_loader_3c = ZTFOutlierLoader(ztf_params, pickles_usage=False)
#
#
#     data_loaders = [
#         # hits_loader_4c,
#         ztf_loader_3c
#     ]
#     # transformer = RankingTransformer()
#     transformer = NoCompositionTransformer()
#
#
#     fid_selector = TransformSelectorRawLogFIDMatrixParallel(
#         verbose=VERBOSE, show=SHOW, from_scratch=FROM_SCRATCH
#     )
#
#     for data_loader_i in data_loaders:
#         print('Original n transforms %i' % transformer.n_transforms)
#         orig_trf = transformer.transformation_tuples[:]
#         x_train = data_loader_i.get_outlier_detection_datasets()[0][0]
#         print(x_train.shape)
#         matrix = fid_selector.get_useful_trfs_matrix(data_loader_i,
#             x_train, transformer)
#         transformer = fid_selector.get_transform_selection_transformer(
#             data_loader_i, x_train, transformer
#         )
#         plt.imshow(matrix)
#         plt.colorbar()
#         plt.show()
#         print('')
#         print('Final n transforms %i' % transformer.n_transforms)
#         transformer.set_transformations_to_perform(orig_trf)