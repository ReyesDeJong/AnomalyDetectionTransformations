import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from parameters import param_keys, general_keys
from modules.data_set_generic import Dataset
from modules.data_loaders.frame_to_input import FrameToInput
from modules.data_splitters.data_splitter_n_samples import DatasetDividerInt
import numpy as np
from modules.geometric_transform import transformations_tf
import time
from modules import utils
import tensorflow as tf
from models.transformer_od_already_transformed import AlreadyTransformODModel


def get_df_dataset_from_name(params: dict, path: str) -> Dataset:
  params_copy = params.copy()
  params_copy.update({
    param_keys.DATA_PATH_TRAIN: path
  })
  frame_to_input = FrameToInput(params_copy)
  frame_to_input.dataset_preprocessor.set_pipeline(
      [frame_to_input.dataset_preprocessor.image_check_single_image,
       frame_to_input.dataset_preprocessor.image_clean_misshaped,
       frame_to_input.dataset_preprocessor.image_select_channels,
       frame_to_input.dataset_preprocessor.image_normalize_by_image_1_1,
       frame_to_input.dataset_preprocessor.image_nan_to_num,
       frame_to_input.dataset_preprocessor.image_crop_at_center,
       frame_to_input.dataset_preprocessor.labels_to_real_bogus
       ])
  return frame_to_input.get_single_dataset()


if __name__ == "__main__":
  random_seed = 42
  val_inlier_percentage = 0.1
  data_name = 'converted_pancho_septiembre.pkl'
  data_folder = "/home/ereyes/Projects/Thesis/datasets/ALeRCE_data"
  save_folder = '/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/OD_retieved_2'
  data_path = os.path.join(data_folder, data_name)

  n_classes = 2
  params = {
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.BATCH_SIZE: None,
    param_keys.N_INPUT_CHANNELS: 3,
    param_keys.CHANNELS_TO_USE: [0, 1, 2],
    param_keys.NANS_TO: 0,
    param_keys.CROP_SIZE: 21,
    param_keys.TEST_SIZE: n_classes * 532,
    param_keys.VAL_SIZE: None,
    param_keys.VALIDATION_RANDOM_SEED: 42,
    param_keys.CONVERTED_DATA_SAVEPATH: None,
    param_keys.BOGUS_LABEL_VALUE: None,
  }

  # normal_data = get_df_dataset_from_name(params, data_path)
  frame_to_input = FrameToInput(params)
  frame_to_input.dataset_preprocessor.set_pipeline(
      [frame_to_input.dataset_preprocessor.image_check_single_image,
       frame_to_input.dataset_preprocessor.image_clean_misshaped,
       frame_to_input.dataset_preprocessor.image_select_channels,
       frame_to_input.dataset_preprocessor.image_normalize_by_image_1_1,
       frame_to_input.dataset_preprocessor.image_nan_to_num,
       frame_to_input.dataset_preprocessor.image_crop_at_center,
       frame_to_input.dataset_preprocessor.labels_to_real_bogus
       ])
  cross_match_dataset = frame_to_input.get_single_dataset()
  print('cross_match data:, ',
        np.unique(cross_match_dataset.data_label, return_counts=True))
  #bogus Alerce
  bogus_data_name = 'bogus_juliano_franz_pancho.pkl'
  bogus_path = os.path.join(
      data_folder, 'converted_' + bogus_data_name)
  bogus_dataset = get_df_dataset_from_name(params, bogus_path)
  print('\n Alerce Bogus.shape: ', bogus_dataset.data_array.shape, '\n')
  #Tns
  tns_data_name = 'tns_confirmed_sn.pkl'
  tns_path = os.path.join(
      data_folder, 'converted_' + tns_data_name)
  tns_dataset = get_df_dataset_from_name(params, tns_path)
  print('\n TNS SNe_(real).shape: ', tns_dataset.data_array.shape, '\n')
  #mergin xmatch+alerce+ashish+tns
  merged_data_array = np.concatenate(
      [cross_match_dataset.data_array, bogus_dataset.data_array,
       tns_dataset.data_array])
  merged_data_label = np.concatenate(
      [cross_match_dataset.data_label, np.zeros(len(bogus_dataset.data_array)),
       np.ones(len(tns_dataset.data_array))])
  merged_dataset = Dataset(data_array=merged_data_array,
                           data_label=merged_data_label, batch_size=None)
  print('merged data: ',
        np.unique(merged_dataset.data_label, return_counts=True))

  #test set extraction
  data_divider = DatasetDividerInt(merged_dataset, test_size=params[param_keys.TEST_SIZE])
  train_dataset_real_bogus, test_dataset =data_divider.get_train_test_data_set_objs()
  print('merged train_real_bogus data: ',
        np.unique(train_dataset_real_bogus.data_label, return_counts=True))
  print('merged test data: ',
        np.unique(test_dataset.data_label, return_counts=True))

  # separating real - bogus in train set
  # 1 real - 0 bogus
  bogus_label_value = 0
  data_bogus = train_dataset_real_bogus .data_array[
    train_dataset_real_bogus.data_label == bogus_label_value]
  data_real = train_dataset_real_bogus.data_array[
    train_dataset_real_bogus.data_label != bogus_label_value]

  data_real_indxs = np.arange(len(data_real))
  np.random.RandomState(seed=random_seed).shuffle(data_real_indxs)
  val_size_inliers = int(
      np.round(len(data_real) * val_inlier_percentage))

  # data splitting
  # TODO: Check
  real_aux = data_real[
    data_real_indxs[len(data_bogus):]]
  real_test = data_real[
    data_real_indxs[:len(data_bogus)]]
  real_train = real_aux[val_size_inliers:]
  real_val = real_aux[:val_size_inliers]

  print('All inliers, except mayor test: ', len(data_real))
  print('Train inliers: ', len(real_train))
  print('Val inliers: ', len(real_val))
  print('Test inliers: ', len(real_test))

  x_train = real_train
  x_val = real_val
  x_test = np.concatenate([real_test, data_bogus])
  y_test = np.concatenate(
      [np.ones(len(real_test)), np.zeros(len(data_bogus))])
  print('y_test: ', np.unique(y_test, return_counts=True))


  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  start_time = time.time()

  transformer = transformations_tf.Transformer()
  x_train_transform, y_train_transform = transformer.apply_all_transforms(
      x=x_train)
  x_val_transform, y_val_transform = transformer.apply_all_transforms(
      x=x_val)
  x_test_transform, y_test_transform = transformer.apply_all_transforms(
      x=x_test)

  mdl = AlreadyTransformODModel(transformer=transformer,
                                input_shape=x_train.shape[1:], name='REAL-BOG')

  batch_size = 128
  mdl.fit(
      x_train_transform, x_val_transform, train_batch_size=batch_size,
      epochs=0  # int(np.ceil(200 / transformer.n_transforms))
  )
  met_dict = mdl.evaluate_od(
      x_train_transform, x_test_transform, y_test, 'new_ztf', 'real',
      x_val_transform, save_hist_folder_path=mdl.specific_model_folder)

  print('\nroc_auc')
  for key in met_dict.keys():
    print(key, met_dict[key]['roc_auc'])
  print(
      "Train and evaluate %s" % utils.timer(
          start_time, time.time()))

  dirichlet_test_pred = met_dict[general_keys.DIRICHLET]['clf']
  print(np.mean(y_test == dirichlet_test_pred))
  #
  # # # saving GEOT_test_Inliers_Test_U_ALL_Bogus_(5)_U_(6)
  # # utils.save_pickle(Dataset(x_test, y_test, batch_size=None),
  # #                   os.path.join(save_folder, 'GEOT_test_Inliers_Test_U_ALL_Bogus_(5)_U_(6)_dataset.pkl'))
  #
  # #Saving Detected_Boguses_(7)
  # bogus_save_path = os.path.join(save_folder, 'Detected_Boguses_(7).pkl')
  # new_od_boguses = x_test[dirichlet_test_pred == bogus_label_value]
  # print('NEW_OD_BOGUSES: ', len(new_od_boguses))
  # utils.save_pickle(new_od_boguses, bogus_save_path)
  #
  # #Saving Train_Geotransform_(3)_U_(4)
  # real_normal_train_val_data_path = os.path.join(save_folder,
  #                                                'Train_Geotransform_(3)_U_(4).pkl')
  # real_normal_train_val_data = np.concatenate([x_train, x_val])
  # utils.save_pickle(real_normal_train_val_data, real_normal_train_val_data_path)

"""

"""
