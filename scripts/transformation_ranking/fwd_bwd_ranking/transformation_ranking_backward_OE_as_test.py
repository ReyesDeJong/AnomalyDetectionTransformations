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

from modules.geometric_transform.transformer_for_ranking import \
    RankingTransformer
from models.transformer_od_simple_net import TransformODSimpleModel
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
from itertools import chain, combinations
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys
import numpy as np
from modules import utils
import tensorflow as tf


def get_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(
        chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def prepare_images(unprepraed_images, images_example):
    if unprepraed_images.shape[1] != images_example.shape[1] or \
        unprepraed_images.shape[2] != images_example.shape[2]:
        unprepraed_images = tf.image.resize(unprepraed_images,
                                            images_example.shape[1:3]).numpy()
    if unprepraed_images.shape[-1] <= images_example.shape[-1]:
        while unprepraed_images.shape[-1] != images_example.shape[-1]:
            unprepraed_images = np.concatenate(
                [unprepraed_images, unprepraed_images[..., -1][..., None]],
                axis=-1)
    else:
        unprepraed_images = unprepraed_images[..., :images_example.shape[-1]]
    unprepraed_images = utils.normalize_by_channel_1_1(unprepraed_images)
    return unprepraed_images


def main():
    MODEL_CHKP_PATH = os.path.join(PROJECT_PATH, 'results',
                                   'Trf_Rank_Bwd')
    RESULT_PATH = 'aux_results'
    utils.check_paths(RESULT_PATH)
    METRIC_TO_RANK_ON = 'roc_auc'
    N_RUNS = 1
    EPOCHS_TO_USE = 1000

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
        # hits_loader,
        ztf_loader
    ]

    for loader_i in data_loaders:
        (x_train, y_train), (x_val, y_val), (
            x_test, y_test) = loader_i.get_outlier_detection_datasets()
        gt_outliers = x_test[y_test != 1]
        inliers = x_test[y_test == 1]
        if loader_i.name == hits_loader.name:
            _, _, (
                x_test_other,
                y_test_other) = ztf_loader.get_outlier_detection_datasets()
            other_set_outliers = x_test_other[y_test_other != 1][
                                 :int(len(x_test) // 2)]
        else:
            hits_params = {
                loader_keys.DATA_PATH: os.path.join(
                    PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
                loader_keys.N_SAMPLES_BY_CLASS: 10000,
                loader_keys.TEST_PERCENTAGE: 0.3,
                loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
                loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
                loader_keys.CROP_SIZE: 21,
                general_keys.RANDOM_SEED: 42,
                loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
            }
            hits_loader = HiTSOutlierLoader(hits_params, pickles_usage=False)
            _, _, (
                x_test_other,
                y_test_other) = hits_loader.get_outlier_detection_datasets()
            other_set_outliers = x_test_other[y_test_other != 1][
                                 :int(len(x_test) // 2)]
        other_set_outliers = prepare_images(other_set_outliers, x_train)

        outliers_dict = {
            # 'gt': gt_outliers,
            # 'shuffle': inliers_shuffle,
            'other_astro': other_set_outliers,
            # 'mnist': mnist,
            # 'cifar10': cifar10
        }

        pickle_name = 'rank_bwd_%s_%s.pkl' % (
            loader_i.name, list(outliers_dict.keys())[0])
        pickle_name = os.path.join(RESULT_PATH, pickle_name)
        trfer = RankingTransformer()
        orig_trf_list = trfer.transformation_tuples
        trf_list = orig_trf_list[:]
        model = TransformODSimpleModel(
            loader_i, trfer, input_shape=x_train.shape[1:],
            results_folder_name=MODEL_CHKP_PATH)
        model.fit(x_train, x_val, epochs=EPOCHS_TO_USE, patience=0)
        outlier_key = list(outliers_dict.keys())[0]
        current_outliers = outliers_dict[outlier_key]
        current_x_test = np.concatenate([inliers, current_outliers], axis=0)
        current_y_test = np.concatenate(
            [y_test[y_test == 1], y_test[y_test != 1]], axis=0)
        results = model.evaluate_od(
            x_train, current_x_test, current_y_test,
            '%s_%s' % (loader_i.name, outlier_key), 'real', x_val)
        print('AUROC using all transformations %i %s_%s %.5f' % (
            len(trf_list), loader_i.name, outlier_key,
            results['dirichlet']['roc_auc']))

        best_rank_metric_run_i = results['dirichlet']['roc_auc']
        best_trf = trf_list
        results_run_i = {}
        for leaving_out_step_i in range(len(trf_list) - 2):
            print(len(trf_list), trf_list)
            transformations_to_leave_out = trf_list[1:]
            best_rank_metric_in_this_leaving_round = 0
            for transformation_i_lo_leave_out in transformations_to_leave_out:
                trf_to_perform = list(trf_list[:])
                trf_to_perform.remove(transformation_i_lo_leave_out)
                trf_to_perform = tuple(trf_to_perform)
                print(len(trf_to_perform), trf_to_perform)
                trfer.set_transformations_to_perform(trf_to_perform)
                model = TransformODSimpleModel(
                    loader_i, trfer, input_shape=x_train.shape[1:],
                    results_folder_name=MODEL_CHKP_PATH)
                model.fit(x_train, x_val, epochs=EPOCHS_TO_USE, patience=0)
                results = model.evaluate_od(
                    x_train, current_x_test, current_y_test,
                    '%s_%s' % (loader_i.name, outlier_key), 'real', x_val)
                print('%i %s_%s %.5f' % (
                    len(trf_to_perform), loader_i.name, outlier_key,
                    results['dirichlet']['roc_auc']))
                model_result_metric = results['dirichlet']['roc_auc']
                if model_result_metric > best_rank_metric_run_i:
                    best_rank_metric_run_i = model_result_metric
                    best_trf = trf_to_perform
                    print('best %i %s' % (len(best_trf), best_trf))
                if model_result_metric > best_rank_metric_in_this_leaving_round:
                    best_rank_metric_in_this_leaving_round = model_result_metric
                    best_trf_in_round = trf_to_perform
                    trf_removed_in_round = transformation_i_lo_leave_out
                    print('best in this round %i %s' % (
                    len(best_trf_in_round), best_trf_in_round))
                results_run_i[transformation_i_lo_leave_out] = [trf_to_perform,
                                                                results]
            print('Transformation removed in round %i %s' % (
            len(best_trf_in_round), trf_removed_in_round))
            trf_list = best_trf_in_round
        print('Best Trf %s_%s %i %s: %f' % (
            loader_i.name, outlier_key, len(best_trf), str(best_trf),
            best_rank_metric_run_i))
        if 'gt' not in outlier_key:
            trfer.set_transformations_to_perform(orig_trf_list)
            model = TransformODSimpleModel(
                loader_i, trfer, input_shape=x_train.shape[1:],
                results_folder_name=MODEL_CHKP_PATH)
            model.fit(x_train, x_val, epochs=EPOCHS_TO_USE, patience=0)
            current_x_test = np.concatenate([inliers, gt_outliers], axis=0)
            current_y_test = np.concatenate(
                [y_test[y_test == 1], y_test[y_test != 1]], axis=0)
            results = model.evaluate_od(
                x_train, current_x_test, current_y_test,
                '%s_%s' % (loader_i.name, outlier_key), 'real', x_val)
            print('Using All Transformations %i %s_%s %.5f' % (
                len(orig_trf_list), loader_i.name, 'gt_outliers',
                results['dirichlet']['roc_auc']))

            trfer.set_transformations_to_perform(best_trf)
            model = TransformODSimpleModel(
                loader_i, trfer, input_shape=x_train.shape[1:],
                results_folder_name=MODEL_CHKP_PATH)
            model.fit(x_train, x_val, epochs=EPOCHS_TO_USE, patience=0)
            results = model.evaluate_od(
                x_train, current_x_test, current_y_test,
                '%s_%s' % (loader_i.name, outlier_key), 'real', x_val)
            print('Using Selected Transformations %i %s_%s %.5f' % (
                len(best_trf), loader_i.name, 'gt_outliers',
                results['dirichlet']['roc_auc']))


if __name__ == "__main__":
    print('')
    main()
