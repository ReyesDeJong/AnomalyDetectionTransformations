import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from parameters import param_keys, general_keys
from modules.data_set_generic import Dataset
from modules.data_loaders.frame_to_input import FrameToInput
import numpy as np
from modules import utils


def get_df_dataset_from_name(params: dict, path: str) -> Dataset:
  params_copy = params.copy()
  params_copy.update({
    param_keys.DATA_PATH_TRAIN: path
  })
  frame_to_input = FrameToInput(params_copy)
  frame_to_input.dataset_preprocessor.set_pipeline(
      [frame_to_input.dataset_preprocessor.check_single_image,
       frame_to_input.dataset_preprocessor.clean_misshaped,
       frame_to_input.dataset_preprocessor.select_channels,
       frame_to_input.dataset_preprocessor.crop_at_center,
       frame_to_input.dataset_preprocessor.normalize_by_image,
       frame_to_input.dataset_preprocessor.nan_to_num,
       frame_to_input.dataset_preprocessor.labels_to_real_bogus
       ])
  return frame_to_input.get_single_dataset()


if __name__ == "__main__":
  random_seed = 42
  # n_bogus_per_dataset = 1500
  # n_inliers_val = 1000
  # n_inliers_train = 7000
  # percentage of inliers, without test inliers
  val_inlier_percentage = 0.1
  data_name = 'converted_pancho_septiembre.pkl'
  data_folder = "/home/ereyes/Projects/Thesis/datasets/ALeRCE_data"
  save_folder = '/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/'
  utils.check_path(save_folder)
  data_path = os.path.join(data_folder, data_name)

  n_classes = 5
  params = {
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.BATCH_SIZE: None,
    param_keys.N_INPUT_CHANNELS: 3,
    param_keys.CHANNELS_TO_USE: [0, 1, 2],
    param_keys.NANS_TO: 0,
    param_keys.CROP_SIZE: 21,
    param_keys.TEST_SIZE: n_classes * 200,
    param_keys.VAL_SIZE: n_classes * 100,
    param_keys.VALIDATION_RANDOM_SEED: random_seed,
    param_keys.CONVERTED_DATA_SAVEPATH: None,
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
  # aux_datasets_dict = frame_to_input.get_datasets()
  # print('Aux Train Set: ',
  #       np.unique(aux_datasets_dict[general_keys.TRAIN].data_label,
  #                 return_counts=True))
  # print('Aux Test Set: ',
  #       np.unique(aux_datasets_dict[general_keys.TEST].data_label,
  #                 return_counts=True))
  # print('Aux Val Set: ',
  #       np.unique(aux_datasets_dict[general_keys.VALIDATION].data_label,
  #                 return_counts=True))
  single_dataset = frame_to_input.get_single_dataset()
  print('all data Set: ', np.unique(single_dataset.data_label,
                                    return_counts=True))

  bogus_data_name = 'bogus_juliano_franz_pancho.pkl'
  bogus_path = os.path.join(
      data_folder, 'converted_' + bogus_data_name)
  bogus_dataset = get_df_dataset_from_name(params, bogus_path)
  print('\n Alerce Bogus.shape: ', bogus_dataset.data_array.shape, '\n')
  real_data_name = 'tns_confirmed_sn.pkl'
  real_path = os.path.join(
      data_folder, 'converted_' + real_data_name)
  tns_dataset = get_df_dataset_from_name(params, real_path)
  print('\n TNS SNe_(real).shape: ', tns_dataset.data_array.shape, '\n')

  # Bogus dataset build
  bogus_ashish = single_dataset.data_array[single_dataset.data_label == 4]
  print('Bogus_ ashish: ', bogus_ashish.shape)
  np.random.RandomState(seed=random_seed).shuffle(bogus_ashish)
  np.random.RandomState(seed=random_seed).shuffle(bogus_dataset.data_array)
  bogus_joint_data = np.concatenate(
      [bogus_dataset.data_array,
       bogus_ashish])
  print('Bogus Test: ', bogus_joint_data.shape)

  # Inliers Test
  # n_inliers_not_sne_test = n_bogus_per_dataset * 2 - len(
  #     tns_dataset.data_array) - (len(
  #     aux_datasets_dict[general_keys.TEST].data_array) / n_classes)
  # extracting inliers without supernovae
  # indxs_not_sne = np.arange(len(single_dataset.data_array))[
  #   single_dataset.data_label != 1]
  indxs_not_bogus = np.arange(len(single_dataset.data_array))[
    single_dataset.data_label != 4]
  # inlier_not_sne_indxs = np.intersect1d(indxs_not_sne, indxs_not_bogus)
  inlier_not_bogus_single_dataset = single_dataset.data_array[indxs_not_bogus]
  labels_not_bogus_single_dataset = single_dataset.data_label[indxs_not_bogus]

  inliers = np.concatenate(
      [inlier_not_bogus_single_dataset, tns_dataset.data_array])
  inliers_labels = np.concatenate(
      [labels_not_bogus_single_dataset,
       np.ones(tns_dataset.data_array.shape[0])])
  print('All Inliers %s, %s' % (
      str(inliers.shape), str(np.unique(inliers_labels, return_counts=True))))

  inlier_indexes = np.arange(len(inliers))
  np.random.RandomState(seed=random_seed).shuffle(inlier_indexes)
  test_inliers = inliers[inlier_indexes[:len(bogus_joint_data)]]
  remaining_inliers = inliers[inlier_indexes[len(bogus_joint_data):]]
  remaining_inlier_indexes = np.arange(len(remaining_inliers))
  np.random.RandomState(seed=random_seed).shuffle(remaining_inlier_indexes)
  val_inliers = remaining_inliers[
                :int(len(remaining_inliers)*val_inlier_percentage)]
  train_inliers = remaining_inliers[
                  int(len(remaining_inliers)*val_inlier_percentage):]
  print('Inliers numbers Train %i Val %i Test %i' % (
      len(train_inliers), len(val_inliers), len(test_inliers)))



  x_test = np.concatenate([test_inliers, bogus_joint_data])
  y_test = np.concatenate(
      [np.ones(len(test_inliers)), np.zeros(len(bogus_joint_data))])
  print('\nTest set: ',
        np.unique(y_test, return_counts=True), '\n')

  x_val = val_inliers
  y_val = np.ones(len(val_inliers))
  x_train = train_inliers
  y_train = np.ones(len(train_inliers))

  new_dataset_tuples = ((x_train, y_train), (x_val, y_val), (
    x_test, y_test))
  utils.save_pickle(
      new_dataset_tuples,
      os.path.join(save_folder, 'all_ztf_dataset_tuples.pkl'))
