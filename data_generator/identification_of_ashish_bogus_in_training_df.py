import os
import sys

"""
From df of training set and ashish_bogus, generate traininset_df without 
ashish_bus df, including idexes of every sample in training df
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
import numpy as np
import pandas as pd
from tqdm import trange


def include_index_as_column(df: pd.DataFrame, column_name='index_col'):
    df[column_name] = df.index
    return df

def check_if_training_df_no_bogus_have_unique_oids(training_df: pd.DataFrame):
    training_df_no_bogus = training_df[training_df['class'] != 'bogus']
    unique_oids_training_df_no_bogus = np.unique(training_df_no_bogus['oid'])
    assert len(training_df_no_bogus) == len(unique_oids_training_df_no_bogus)

def get_sample_str(df: pd.DataFrame, sample_i, column_names):
    sample = df.iloc[sample_i]
    string = ''
    for col_name in column_names:
        string += str(sample[col_name])
    return string

def get_df_with_data_source_indicated(training_df_path, ashish_bogus_df_path)->pd.DataFrame:
    training_df = pd.read_pickle(training_df_path)
    ashish_bogus_df = pd.read_pickle(ashish_bogus_df_path)[training_df.columns]
    check_if_training_df_no_bogus_have_unique_oids(training_df)
    training_df_bogus = training_df[training_df['class'] == 'bogus']
    alerce_bogus_in_training_df = pd.concat(
        [ashish_bogus_df, training_df_bogus]).drop_duplicates(keep=False)
    ashish_bogus_in_training_df = pd.concat(
        [alerce_bogus_in_training_df,
         training_df_bogus]).drop_duplicates(keep=False)
    # check if ashish_bogus and ashish_bogus_in_training_set are in same order and equal
    equals_list = []
    for i in trange(len(ashish_bogus_df)):
        equals_list.append(
            ashish_bogus_df.iloc[i].equals(ashish_bogus_in_training_df.iloc[i]))
    # print(np.mean(equals_list))
    assert np.mean(equals_list) == 1.0
    # new df with data source indicated
    training_df_no_bogus = training_df[training_df['class'] != 'bogus']
    training_df_no_bogus['data_source'] = 'alerce'
    alerce_bogus_in_training_df['data_source'] = 'alerce'
    ashish_bogus_in_training_df['data_source'] = 'ashish'
    # print(len(training_df))
    # print(len(training_df_no_bogus) + len(alerce_bogus_in_training_df) + len(
    #     ashish_bogus_in_training_df))
    assert len(training_df) == (
            len(training_df_no_bogus) + len(alerce_bogus_in_training_df) + len(
            ashish_bogus_in_training_df))
    df_with_clear_source = pd.concat(
        [training_df_no_bogus, alerce_bogus_in_training_df,
         ashish_bogus_in_training_df]).drop_duplicates(keep=False).sort_index()
    # print(len(df_with_clear_source))
    assert len(df_with_clear_source) == len(training_df)
    df_with_clear_source = include_index_as_column(df_with_clear_source)

    return df_with_clear_source



if __name__ == "__main__":
    data_name = 'training_set_Aug-07-2020.pkl'
    data_folder = "/home/ereyes/Projects/Thesis/stamp_classifier_updater/data/"
    data_path = os.path.join(data_folder, data_name)
    training_df_path = data_path#os.path.join(PROJECT_PATH, "../pickles", 'training_set_May-06-2020.pkl')
    ashish_bogus_df_path = os.path.join(data_folder, 'bogus_ashish.pkl')
    training_df = pd.read_pickle(training_df_path)

    df_with_clear_source = get_df_with_data_source_indicated(
        training_df_path, ashish_bogus_df_path)

    print(df_with_clear_source.tail())
    print('df_with_clear_source ',
          np.unique(df_with_clear_source['data_source'], return_counts=True))
    print('df_with_clear_source ',
          np.unique(df_with_clear_source['class'], return_counts=True))
    print('training_df ',
          np.unique(training_df['class'], return_counts=True))

    # OLD way to do things by brute force
    # # get ashish_bogus in training_bogus
    # columns_to_compare = ['oid', 'cutoutScience', 'cutoutTemplate', 'cutoutDifference']
    #
    # ashish_bogus_in_training_df = []
    # exclusive_bogus_in_training_df = []
    # # get ashish_bogus in training_bogus
    # for ashish_i in trange(len(ashish_bogus_df)):
    #     ashish_str = get_sample_str(ashish_bogus_df, ashish_i, columns_to_compare)
    #     matches_found = 0
    #     for train_bogus_i in range(len(training_df_bogus)):
    #         train_bogus_str = get_sample_str(training_df_bogus, train_bogus_i, columns_to_compare)
    #         if train_bogus_str == ashish_str:
    #             ashish_bogus_in_training_df.append(training_df_bogus.iloc[train_bogus_i])
    #             matches_found +=1
    #             assert matches_found == 1
    #
    # ashish_bogus_in_training_df = pd.DataFrame(ashish_bogus_in_training_df)
    # print(len(ashish_bogus_in_training_df))



