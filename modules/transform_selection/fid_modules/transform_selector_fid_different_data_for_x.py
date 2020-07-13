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
from modules.transform_selection.fid_modules.transform_selector_fid import TransformSelectorFRawLogFID
import numpy as np
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from modules import utils
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from parameters import loader_keys, general_keys


class TransformSelectorFRawLogFIDOtherDatAsX(TransformSelectorFRawLogFID):

    def __init__(self, threshold_magnitud_order=1):
        super().__init__()
        self._generator_name = 'RawLogFIDOtherAsX'
        self.threshold_magnitud_order = threshold_magnitud_order

    def _get_distance_T0_between_orig_and_other_data(self, transformer,
        moments_original, x_other):
        features_trf = \
            transformer.apply_transforms(x_other, [0])[0]
        moments_trf = self._get_fid_moments_from_data(features_trf)
        dist_value = self._get_distance_original_trf(moments_original,
                                                     moments_trf)
        return dist_value

    def _create_dict_with_scores_for_each_transform(
        self, x_dataset, transformer: AbstractTransformer):
        n_samples_in_x = len(x_dataset)
        x_dataset_T0 = x_dataset[:n_samples_in_x//2]
        x_dataset_Ti = x_dataset[n_samples_in_x // 2:]
        # doesn't include score on T0
        self._dict_trf_scores = {}
        x_dataset_T0 = \
            transformer.apply_transforms(x_dataset_T0, [0])[0]
        moments_original = self._get_fid_moments_from_data(x_dataset_T0)
        dist_to_dist_between_T0_and_T0_from_other = \
            self._get_distance_T0_between_orig_and_other_data(
            transformer, moments_original, x_dataset_Ti)
        n_transforms = transformer.n_transforms
        transformations_idxs_to_evaluate = np.arange(n_transforms)[1:]
        for transform_idx_i in tqdm(
            range(len(transformations_idxs_to_evaluate)),
            disable=not self.verbose):
            transform_idx = transformations_idxs_to_evaluate[transform_idx_i]
            features_trf = \
                transformer.apply_transforms(x_dataset_Ti, [transform_idx])[0]
            moments_trf = self._get_fid_moments_from_data(features_trf)
            dist_value = self._get_distance_original_trf(moments_original,
                                                         moments_trf)
            dist_value = np.abs(
                dist_value-dist_to_dist_between_T0_and_T0_from_other)
            trf_name = transform_idx
            transformation_tuple = transformer.transformation_tuples[
                transform_idx]
            self._dict_trf_scores[trf_name] = {
                general_keys.TRANSFORMATION_TUPLE: transformation_tuple,
                general_keys.SCORE: dist_value
            }

    def _cluster_fid_scores_in_two_groups(self, transformation_scores):
        y_thr = transformation_scores > self.threshold_magnitud_order
        self.plot_clusters(transformation_scores, y_thr)
        return y_thr

if __name__ == "__main__":
    from modules.geometric_transform.transformer_for_ranking import \
        RankingTransformer
    from modules.geometric_transform.transformations_tf import PlusKernelTransformer, Transformer
    from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader

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
        loader_keys.USED_CHANNELS: [2],  # [0, 1, 2, 3],  #
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    hits_params.update({loader_keys.USED_CHANNELS: [0, 1, 2, 3]})
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

    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 10000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
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

    data_loaders = [
        # ztf_loader,
        # hits_loader,
        # hits_loader_4c,
        ztf_loader_3c
    ]
    # transformer = RankingTransformer()
    transformer = PlusKernelTransformer()
    # transformer = Transformer()
    print('Original n transforms %i' % transformer.n_transforms)

    fid_selector = TransformSelectorFRawLogFIDOtherDatAsX()

    for data_loader_i in data_loaders:
        x_train = data_loader_i.get_outlier_detection_datasets()[0][0]
        print(x_train.shape)
        selected_trfs = fid_selector.get_selected_transformations(
            x_train, transformer, verbose=VERBOSE, show=SHOW)
        print('%i %s' % (len(selected_trfs), str(selected_trfs)))
        print('')
    print('Original n transforms %i' % transformer.n_transforms)
