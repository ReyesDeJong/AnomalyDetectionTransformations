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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from parameters import loader_keys, general_keys


class TransformSelectorFRawLogFID(object):

    def __init__(self):
        self._generator_name = 'RawLogFID'
        self._dict_trf_scores = None
        self.verbose = False
        self.show = False

    def _get_fid_moments_from_data(self, data):
        mu, sigma = fid. \
            calculate_activation_statistics_from_activation_array(data)
        return (mu, sigma)

    def _get_distance_original_trf(self, fid_moments_original, fid_moments_trf):
        fid_value = fid.calculate_frechet_distance(
            fid_moments_original[0], fid_moments_original[1],
            fid_moments_trf[0],
            fid_moments_trf[1])
        log_fid_value = np.log(fid_value)
        return log_fid_value

    def _create_dict_with_scores_for_each_transform(
        self, x_dataset, transformer: AbstractTransformer):
        # doesn't include score on T0
        self._dict_trf_scores = {}
        x_dataset = \
            transformer.apply_transforms(x_dataset, [0])[0]
        moments_original = self._get_fid_moments_from_data(x_dataset)
        n_transforms = transformer.n_transforms
        transformations_idxs_to_evaluate = np.arange(n_transforms)[1:]
        for transform_idx_i in tqdm(
            range(len(transformations_idxs_to_evaluate)),
            disable=not self.verbose):
            transform_idx = transformations_idxs_to_evaluate[transform_idx_i]
            features_trf = \
                transformer.apply_transforms(x_dataset, [transform_idx])[0]
            moments_trf = self._get_fid_moments_from_data(features_trf)
            dist_value = self._get_distance_original_trf(moments_original,
                                                         moments_trf)
            trf_name = transform_idx
            transformation_tuple = transformer.transformation_tuples[
                transform_idx]
            self._dict_trf_scores[trf_name] = {
                general_keys.TRANSFORMATION_TUPLE: transformation_tuple,
                general_keys.SCORE: dist_value
            }

    def get_all_transformation_tuples_and_scores(
        self, x_dataset, transformer: AbstractTransformer):
        self._create_dict_with_scores_for_each_transform(
            x_dataset, transformer)
        transformation_indexes = np.sort(list(self._dict_trf_scores.keys()))
        transformation_tuples = []
        transformation_scores = []
        for transformation_indexes_i in transformation_indexes:
            transformation_tuples.append(self._dict_trf_scores[
                                             transformation_indexes_i][
                                             general_keys.TRANSFORMATION_TUPLE])
            transformation_scores.append(self._dict_trf_scores[
                                             transformation_indexes_i][
                                             general_keys.SCORE])
        return transformation_tuples, np.array(transformation_scores)

    def plot_clusters(self, transformation_scores, y_labels):
        if self.show:
            print('Transformation scores:', transformation_scores)
            plt.plot([1] * len(transformation_scores),
                     transformation_scores[:, 0],
                     'o')
            plt.show()
            plt.scatter([1] * len(transformation_scores),
                        transformation_scores[:, 0], c=y_labels)
            plt.show()

    def _cluster_fid_scores_in_two_groups(self, transformation_scores):
        transformation_scores = transformation_scores[..., None]
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(transformation_scores)
        y_kmeans = kmeans.predict(transformation_scores)
        self.plot_clusters(transformation_scores, y_kmeans)
        return y_kmeans

    def _get_usefull_transformations(self, transformation_tuples,
        transformations_scores, transformation_predictions):
        transformation_aux_idxs = np.arange(len(transformations_scores))
        pred_labeles = np.unique(transformation_predictions)
        cluster_1_idexes = transformation_aux_idxs[
            transformation_predictions == pred_labeles[0]]
        cluster_2_idexes = transformation_aux_idxs[
            transformation_predictions == pred_labeles[1]]
        max_cluster_1_score = np.max(transformations_scores[cluster_1_idexes])
        max_cluster_2_score = np.max(transformations_scores[cluster_2_idexes])
        if max_cluster_2_score > max_cluster_1_score:
            selected_transforms_idexes = cluster_2_idexes
        else:
            selected_transforms_idexes = cluster_1_idexes
        selected_transform_tuples = np.array(transformation_tuples)[
            selected_transforms_idexes].tolist()
        eliminted_transforms_idexes = \
            list(set(transformation_aux_idxs.tolist()).
                 difference(set(selected_transforms_idexes.tolist())))
        eliminated_transform_tuple = np.array(transformation_tuples)[
            eliminted_transforms_idexes].tolist()
        if self.verbose:
            print('eliminated tuples %i %s' % (
                len(eliminated_transform_tuple),
                str(eliminated_transform_tuple)))
        return selected_transform_tuples

    def get_selected_transformations(
        self, x_dataset, transformer: AbstractTransformer, verbose=False,
        show=False):
        self.verbose = verbose
        self.show = show
        transformation_tuples, transformation_scores = \
            self.get_all_transformation_tuples_and_scores(
                x_dataset, transformer)
        transformations_predictions = self._cluster_fid_scores_in_two_groups(
            transformation_scores)
        useful_transforms = self._get_usefull_transformations(
            transformation_tuples, transformation_scores,
            transformations_predictions)
        selected_trfs = [list(
            transformer.transformation_tuples[0])] + useful_transforms
        return selected_trfs


if __name__ == "__main__":
    from modules.geometric_transform.transformations_tf import \
        PlusKernelTransformer
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

    fid_selector = TransformSelectorFRawLogFID()

    for data_loader_i in data_loaders:
        x_train = data_loader_i.get_outlier_detection_datasets()[0][0]
        print(x_train.shape)
        selected_trfs = fid_selector.get_selected_transformations(
            x_train, transformer, verbose=VERBOSE, show=SHOW)
        print('%i %s' % (len(selected_trfs), str(selected_trfs)))
        print('')
    print('Original n transforms %i' % transformer.n_transforms)
