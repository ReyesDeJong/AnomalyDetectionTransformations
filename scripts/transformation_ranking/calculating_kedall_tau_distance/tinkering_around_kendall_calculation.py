"""
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
import pandas as pd
from modules.utils import get_files_in_path
from scripts.transformation_ranking.kendal_tau_distance_tinkering import kendall_tau_distance

def get_all_transformations_by_dataset_name(
    dataset_name):
    lists_path = os.path.join(
      PROJECT_PATH,
        'scripts/transformation_ranking/'
        'calculating_kedall_tau_distance/top10_lists')
    file_names = get_files_in_path(lists_path)
    dataset_file_names = [file_i for file_i in
                          file_names if dataset_name in file_i]
    # print(dataset_file_names)
    transformations = None
    for file_i in dataset_file_names:
        file_path = os.path.join(lists_path, file_i)
        transform_i = np.concatenate(pd.read_pickle(file_path))
        if transformations is None:
            transformations = transform_i
            continue
        # print(transform_i)
        transformations = np.concatenate([transformations,transform_i], axis=0)
    # print(transformations.shape)
    return transformations

def get_index_of_transformation_in_unique_list(unique_list, transformation):
    idex_list = [i for i, v in enumerate(unique_list) if (transformation==v).all()]
    return idex_list[0]

def get_index_of_transformation_in_unique_list_2(unique_list, transformation):
    lists = []
    for i, v in enumerate(unique_list):
        try:
            if (transformation == v).all():
                lists.append(i)
        except:
            if transformation == v:
                lists.append(i)
    idex_list = lists
    # if len(idex_list)==0:
    #     return 0
    return idex_list[0]

def get_transformations_dict(dataset_name):
    lists_path = os.path.join(
        PROJECT_PATH,
        'scripts/transformation_ranking/'
        'calculating_kedall_tau_distance/top10_lists')
    file_names = get_files_in_path(lists_path)
    dataset_file_names = [file_i for file_i in
                          file_names if dataset_name in file_i]
    # print(dataset_file_names)
    result_dict = {}
    for file_i in dataset_file_names:
        file_path = os.path.join(lists_path, file_i)
        transform_i = pd.read_pickle(file_path)
        trf_origin_name = file_i.split('_')[1].split('.')[0]
        result_dict[trf_origin_name] = transform_i
    # print(result_dict)
    return result_dict

def trf_dicts_trf_to_numbers(trf_dict, unique_transforms):
    trf_dict_as_number = {}
    for key in trf_dict.keys():
        top10_trfs_lists = trfs_dict[key]
        trf_lists_i_as_numbers = []
        for trf_list_i in top10_trfs_lists:
            trf_as_numbers = []
            for trf_i in trf_list_i:
                # print(unique_transforms)
                # print(trf_i)
                index_of_trf_i = get_index_of_transformation_in_unique_list(
                    unique_transforms, trf_i)
                # print(index_of_trf_i)
                trf_as_numbers.append(index_of_trf_i)
            trf_lists_i_as_numbers.append(trf_as_numbers)
        trf_dict_as_number[key] = trf_lists_i_as_numbers
    return trf_dict_as_number

def get_all_kendall_distances(trf_dict_numbers):
    gt_trfs = trf_dict_numbers['gt']
    trfs_set_name_i = trf_dict_numbers['shuffle']

    kendalls_dict = {}
    for key in trf_dict_numbers.keys():
        if key in 'gt':
            continue
        all_kendalls = []
        for trfs_i in trfs_set_name_i:
            kendaldistances_i = []
            for gt_trf_i in gt_trfs:
                # print(gt_trf_i)
                # print(trfs_i)
                intersection = list(set(gt_trf_i) & set(trfs_i))
                # print(intersection)
                intersection_array = np.array(intersection)
                # np.argwhere(intersection_array==0)[0][0]
                a = [np.argwhere(intersection_array==i)[0][0]+1 for i in gt_trf_i if i in intersection]
                b = [np.argwhere(intersection_array==i)[0][0]+1 for i in trfs_i if i in intersection]
                # print(a)
                # print(b)

                kendaldistances_i.append(kendall_tau_distance(a, b))
                # print('')
            all_kendalls.append(np.mean(kendaldistances_i))
        kendalls_dict[key] = np.mean(all_kendalls)
    return kendalls_dict

def get_trfs_numbers_as_number(trf_dict_numbers):
    trfs_number_sets = []
    for key in trf_dict_numbers:
        for trfs_i in trf_dict_numbers[key]:
            trfs_number_sets.append(trfs_i)
    trfs_number_sets_len_3 = [i for i in trfs_number_sets if len(i)==3]
    trfs_number_sets_len_2 = [i for i in trfs_number_sets if len(i) == 2]
    uniques_3 = np.unique(trfs_number_sets_len_3, axis=0)
    uniques_2 = np.unique(trfs_number_sets_len_2, axis=0)
    uniques = list(uniques_3)+list(uniques_2)
    # print(uniques)
    results_dict = {}
    for key in trf_dict_numbers:
        trfs_as_number = []
        for trfs_i in trf_dict_numbers[key]:
            idex_i = get_index_of_transformation_in_unique_list_2(uniques, trfs_i)
            trfs_as_number.append(idex_i)
        results_dict[key] = trfs_as_number
    # print(results_dict)
    return results_dict

def get_kendall_distance_transformations_numbers_lists(trfs_lists_as_numbers):
    gt_trfs = trfs_lists_as_numbers['gt']
    trfs_set_name_i = trfs_lists_as_numbers['shuffle']

    kendalls_dict = {}
    for key in trfs_lists_as_numbers.keys():
        if key in 'gt':
            continue
        trfs_i = trfs_lists_as_numbers[key]
        # print(gt_trfs)
        # print(trfs_i)
        intersection = list(set(gt_trfs) & set(trfs_i))
        # print(intersection)
        intersection_array = np.array(intersection)
        # np.argwhere(intersection_array==0)[0][0]
        a = [np.argwhere(intersection_array==i)[0][0]+1 for i in gt_trfs if i in intersection]
        b = [np.argwhere(intersection_array==i)[0][0]+1 for i in trfs_i if i in intersection]
        # print(a)
        # print(b)
        kendall = kendall_tau_distance(a,b)
        # print(kendall)
        kendalls_dict[key] = np.mean(kendall)
    # print(kendalls_dict)
    return kendalls_dict


if __name__ == "__main__":
    dataset_name = 'ztf'
    # transformations to numbers
    transformations = get_all_transformations_by_dataset_name(dataset_name)
    unique_transforms = list(np.unique(transformations, axis=0))
    trfs_dict = get_transformations_dict(dataset_name)
    # print(trfs_dict['other'].shape)
    trfs_lists_as_numbers = trf_dicts_trf_to_numbers(trfs_dict, unique_transforms)
    # print(trfs_dict)
    # print(trf_dict_numbers)
    kendall_dict = get_all_kendall_distances(trfs_lists_as_numbers)
    print(kendall_dict)
    trfs_lists_as_numbers = get_trfs_numbers_as_number(trfs_lists_as_numbers)
    kendalls_2 = get_kendall_distance_transformations_numbers_lists(trfs_lists_as_numbers)
    print(kendalls_2)

        #
        # print(transformations[1,:])
        # print(unique_transforms)
        # print([i for i, v in enumerate(unique_transforms) if (transformations[1,:]==v).all()])
