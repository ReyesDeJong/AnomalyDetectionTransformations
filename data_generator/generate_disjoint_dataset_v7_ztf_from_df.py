import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from parameters import param_keys
from modules.data_loaders.frame_to_input import FrameToInput
import numpy as np
from modules import utils
from modules.data_set_generic import Dataset
import pandas as pd
from tqdm import tqdm


def get_intersection_idexes(data_array_1, data_array_2):
    idexes_1 = np.arange(len(data_array_1))
    idexes_2 = np.arange(len(data_array_2))
    intersection_idexes_1 = []
    intersection_idexes_2 = []
    for idex_i_1 in tqdm(idexes_1):
        for idex_i_2 in idexes_2:
            if (data_array_1[idex_i_1] == data_array_2[idex_i_2]).all():
                if idex_i_1 in intersection_idexes_1:
                    print('\nidex_1_repeated')
                    print(idex_i_1)
                    print(idex_i_2)
                if idex_i_2 in intersection_idexes_2:
                    print('\nidex_2_repeated')
                    print(idex_i_1)
                    print(idex_i_2)
                # assert idex_i_1 not in intersection_idexes_1
                # assert idex_i_2 not in intersection_idexes_2
                intersection_idexes_1.append(idex_i_1)
                intersection_idexes_2.append(idex_i_2)
    return intersection_idexes_1, intersection_idexes_2


def generate_inliers_outliers_tuples(stamps, labels, random_seed,
    val_inlier_percentage, save_path=None):
    bogus_stamps = stamps[labels == 4]
    inliers_stamps = stamps[labels != 4]

    inlier_indexes = np.arange(len(inliers_stamps))
    np.random.RandomState(seed=random_seed).shuffle(inlier_indexes)
    test_inliers = inliers_stamps[inlier_indexes[:len(bogus_stamps)]]
    remaining_inliers_original_idexes = inlier_indexes[len(bogus_stamps):]
    remaining_inliers = inliers_stamps[inlier_indexes[len(bogus_stamps):]]
    val_inliers = remaining_inliers[
                  :int(len(remaining_inliers) * val_inlier_percentage)]
    train_inliers = remaining_inliers[
                    int(len(remaining_inliers) * val_inlier_percentage):]
    train_inliers_labels = labels[
        remaining_inliers_original_idexes[
        int(len(remaining_inliers) * val_inlier_percentage):
        ]]
    print('\nTrain set: ',
          np.unique(train_inliers_labels, return_counts=True), '\n')
    print('Inliers numbers Train %i Val %i Test %i' % (
        len(train_inliers), len(val_inliers), len(test_inliers)))

    x_test = np.concatenate([test_inliers, bogus_stamps])
    y_test = np.concatenate(
        [np.ones(len(test_inliers)), np.zeros(len(bogus_stamps))])
    print('\nTest set: ',
          np.unique(y_test, return_counts=True), '\n')

    x_val = val_inliers
    y_val = np.ones(len(val_inliers))
    x_train = train_inliers
    y_train = np.ones(len(train_inliers))

    v7_disjoint_dataset_tuples = ((x_train, y_train), (x_val, y_val), (
        x_test, y_test))
    if save_path:
        utils.save_pickle(
            v7_disjoint_dataset_tuples, save_path)
    return v7_disjoint_dataset_tuples


if __name__ == "__main__":
    random_seed = 42
    n_samples_max_under_sample = 10000
    val_inlier_percentage = 0.1
    data_name = 'training_set_Aug-07-2020.pkl'
    data_folder = "/home/ereyes/Projects/Thesis/stamp_classifier_updater/data/"
    small_data_folder = '/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/'
    save_folder = '/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/ztf_v7/'
    disjoint_dataset_path = os.path.join(
        small_data_folder, 'new_small_od_dataset_tuples.pkl')
    ((x_train, y_train), (x_val, y_val), (
        x_test, y_test)) = pd.read_pickle(disjoint_dataset_path)
    print('Small test set: ', np.unique(y_test, return_counts=True))

    converted_ztf_v7_path = '/home/ereyes/Projects/Thesis/datasets/' \
                            'ALeRCE_data/ztf_v7/converted_data.pkl'
    if os.path.exists(converted_ztf_v7_path):
        # dataset_dict = pd.read_pickle(converted_ztf_v7_path)
        # single_dataset = Dataset(dataset_dict["images"], dataset_dict["labels"],
        #                          None)
        data_path = os.path.join(converted_ztf_v7_path)
    else:
        data_path = os.path.join(data_folder, data_name)

    n_classes = 5
    params = {
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.BATCH_SIZE: None,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.NANS_TO: 0,
        param_keys.CROP_SIZE: 21,
        param_keys.TEST_SIZE: n_classes * 300,
        param_keys.VAL_SIZE: n_classes * 100,
        param_keys.VALIDATION_RANDOM_SEED: random_seed,
        param_keys.CONVERTED_DATA_SAVEPATH: converted_ztf_v7_path,
        param_keys.BOGUS_LABEL_VALUE: None,
    }

    # normal_data = get_df_dataset_from_name(params, data_path)
    frame_to_input = FrameToInput(params)
    frame_to_input.dataset_preprocessor.set_pipeline(
        [frame_to_input.dataset_preprocessor.check_single_image,
         frame_to_input.dataset_preprocessor.clean_misshaped,
         frame_to_input.dataset_preprocessor.select_channels,
         frame_to_input.dataset_preprocessor.crop_at_center,
         frame_to_input.dataset_preprocessor.normalize_by_image,
         frame_to_input.dataset_preprocessor.nan_to_num,
         ])
    single_dataset = frame_to_input.get_single_dataset()
    print('ztf v7 all data Set: ', np.unique(single_dataset.data_label,
                                             return_counts=True))
    intersect_indices_1_path = os.path.join(save_folder,
                                            'disjoint_inter_indices_1.pkl')
    intersect_indices_2_path = os.path.join(save_folder,
                                            'disjoint_inter_indices_2.pkl')
    if os.path.exists(intersect_indices_1_path) and os.path.exists(
        intersect_indices_2_path):
        intersection_idexes_1 = pd.read_pickle(intersect_indices_1_path)
        intersection_idexes_2 = pd.read_pickle(intersect_indices_2_path)
    else:
        intersection_idexes_1, intersection_idexes_2 = get_intersection_idexes(
            x_test, single_dataset.data_array)
        utils.save_pickle(
            intersection_idexes_1,
            os.path.join(save_folder, 'disjoint_inter_indices_1.pkl')
        )
        utils.save_pickle(
            intersection_idexes_2,
            os.path.join(save_folder, 'disjoint_inter_indices_2.pkl'))
    print(intersection_idexes_1)
    print(intersection_idexes_2)

    v7_stamps_not_in_test = single_dataset.data_array[
        [i for i in range(len(single_dataset.data_array)) if
         i not in intersection_idexes_2]]
    v7_labels_not_in_test = single_dataset.data_label[
        [i for i in range(len(single_dataset.data_label)) if
         i not in intersection_idexes_2]]

    print('Disjoint Data v7: ',
          np.unique(v7_labels_not_in_test, return_counts=True))

    #------------------
    #v7 ztf disjoint dataset
    # ------------------
    save_path_ztf_v7 = os.path.join(save_folder, 'v7_ztf_disjoint_test.pkl')
    ztf_v7_data_tuples = generate_inliers_outliers_tuples(
        v7_stamps_not_in_test, v7_labels_not_in_test, random_seed,
        val_inlier_percentage, save_path_ztf_v7)

    #------------------
    #v7 ztf disjoint dataset oversampled
    # ------------------
    print('Disjoint Data v7: ',
          np.unique(v7_labels_not_in_test, return_counts=True))
    stamps = v7_stamps_not_in_test
    labels = v7_labels_not_in_test
    bogus_stamps = stamps[labels == 4]
    inliers_stamps = stamps[labels != 4]
    inliers_labels = labels[labels != 4]

    inlier_indexes = np.arange(len(inliers_stamps))
    np.random.RandomState(seed=random_seed).shuffle(inlier_indexes)
    test_inliers = inliers_stamps[inlier_indexes[:len(bogus_stamps)]]
    remaining_inliers_original_idexes = inlier_indexes[len(bogus_stamps):]
    remaining_inliers = inliers_stamps[inlier_indexes[len(bogus_stamps):]]
    remaining_inliers_labels = inliers_labels[
        inlier_indexes[len(bogus_stamps):]]
    print('Disjoint remaining Data v7: ',
          np.unique(remaining_inliers_labels, return_counts=True))
    oversamples_inliers_dataset = Dataset(
        remaining_inliers, remaining_inliers_labels, None)
    oversamples_inliers_dataset.balance_data_by_replication()
    print('Oversampled Data v7 inliers: ',
              np.unique(
                  oversamples_inliers_dataset.data_label, return_counts=True))
    remaining_inliers = oversamples_inliers_dataset.data_array
    val_inliers = remaining_inliers[
                  :int(len(remaining_inliers) * val_inlier_percentage)]
    train_inliers = remaining_inliers[
                    int(len(remaining_inliers) * val_inlier_percentage):]
    train_inliers_labels = labels[
        remaining_inliers_original_idexes[
        int(len(remaining_inliers) * val_inlier_percentage):
        ]]
    print('\nTrain set: ',
          np.unique(train_inliers_labels, return_counts=True), '\n')
    print('Inliers numbers Train %i Val %i Test %i' % (
        len(train_inliers), len(val_inliers), len(test_inliers)))

    x_test = np.concatenate([test_inliers, bogus_stamps])
    y_test = np.concatenate(
        [np.ones(len(test_inliers)), np.zeros(len(bogus_stamps))])
    print('\nTest set: ',
          np.unique(y_test, return_counts=True), '\n')

    x_val = val_inliers
    y_val = np.ones(len(val_inliers))
    x_train = train_inliers
    y_train = np.ones(len(train_inliers))

    v7_disjoint_dataset_oversampled_tuples = ((x_train, y_train), (x_val, y_val), (
        x_test, y_test))
    save_path = os.path.join(save_folder, 'v7_ztf_oversamples_disjoint_test.pkl')
    utils.save_pickle(
        v7_disjoint_dataset_oversampled_tuples, save_path)
    print(np.mean(ztf_v7_data_tuples[2][0]==x_test))

    #------------------
    #v7 ztf disjoint dataset undersampled
    # ------------------
    print('Disjoint Data v7: ',
          np.unique(v7_labels_not_in_test, return_counts=True))
    stamps = v7_stamps_not_in_test
    labels = v7_labels_not_in_test
    bogus_stamps = stamps[labels == 4]
    inliers_stamps = stamps[labels != 4]
    inliers_labels = labels[labels != 4]

    inlier_indexes = np.arange(len(inliers_stamps))
    np.random.RandomState(seed=random_seed).shuffle(inlier_indexes)
    test_inliers = inliers_stamps[inlier_indexes[:len(bogus_stamps)]]
    remaining_inliers_original_idexes = inlier_indexes[len(bogus_stamps):]
    remaining_inliers = inliers_stamps[inlier_indexes[len(bogus_stamps):]]
    remaining_inliers_labels = inliers_labels[
        inlier_indexes[len(bogus_stamps):]]
    print('Disjoint remaining Data v7: ',
          np.unique(remaining_inliers_labels, return_counts=True))
    oversamples_inliers_dataset = Dataset(
        remaining_inliers, remaining_inliers_labels, None)
    oversamples_inliers_dataset.balance_data_by_replication()
    print('Oversampled Data v7 inliers: ',
              np.unique(
                  oversamples_inliers_dataset.data_label, return_counts=True))
    remaining_inliers = oversamples_inliers_dataset.data_array
    val_inliers = remaining_inliers[
                  :int(len(remaining_inliers) * val_inlier_percentage)]
    train_inliers = remaining_inliers[
                    int(len(remaining_inliers) * val_inlier_percentage):]
    train_inliers_labels = labels[
        remaining_inliers_original_idexes[
        int(len(remaining_inliers) * val_inlier_percentage):
        ]]
    print('\nTrain set: ',
          np.unique(train_inliers_labels, return_counts=True), '\n')
    print('Inliers numbers Train %i Val %i Test %i' % (
        len(train_inliers), len(val_inliers), len(test_inliers)))

    x_test = np.concatenate([test_inliers, bogus_stamps])
    y_test = np.concatenate(
        [np.ones(len(test_inliers)), np.zeros(len(bogus_stamps))])
    print('\nTest set: ',
          np.unique(y_test, return_counts=True), '\n')

    x_val = val_inliers
    y_val = np.ones(len(val_inliers))
    x_train = train_inliers
    y_train = np.ones(len(train_inliers))

    v7_disjoint_dataset_oversampled_tuples = ((x_train, y_train), (x_val, y_val), (
        x_test, y_test))
    save_path = os.path.join(save_folder, 'v7_ztf_oversamples_disjoint_test.pkl')
    utils.save_pickle(
        v7_disjoint_dataset_oversampled_tuples, save_path)
    print(np.mean(ztf_v7_data_tuples[2][0]==x_test))




