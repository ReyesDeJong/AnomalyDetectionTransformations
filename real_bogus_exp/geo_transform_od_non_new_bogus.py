import os
import sys


"""
Bogus used ar only Ashish's
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from parameters import param_keys, general_keys
from modules.data_set_generic import Dataset
from modules.data_loaders.frame_to_input import FrameToInput
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
      [frame_to_input.dataset_preprocessor.check_single_image,
       frame_to_input.dataset_preprocessor.clean_misshaped,
       frame_to_input.dataset_preprocessor.select_channels,
       frame_to_input.dataset_preprocessor.normalize_by_image,
       frame_to_input.dataset_preprocessor.nan_to_num,
       frame_to_input.dataset_preprocessor.crop_at_center,
       frame_to_input.dataset_preprocessor.labels_to_real_bogus
       ])
  return frame_to_input.get_single_dataset()


if __name__ == "__main__":
  random_seed = 42
  val_inlier_percentage = 0.1
  data_name = 'converted_pancho_septiembre.pkl'
  data_folder = "/home/ereyes/Projects/Thesis/datasets/ALeRCE_data"
  data_path = os.path.join(data_folder, data_name)

  n_classes = 2
  params = {
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.BATCH_SIZE: None,
    param_keys.N_INPUT_CHANNELS: 3,
    param_keys.CHANNELS_TO_USE: [0, 1, 2],
    param_keys.NANS_TO: 0,
    param_keys.CROP_SIZE: 21,
    param_keys.TEST_SIZE: n_classes * 390,
    param_keys.VAL_SIZE: n_classes * 200,
    param_keys.VALIDATION_RANDOM_SEED: 42,
    param_keys.CONVERTED_DATA_SAVEPATH: None,
    param_keys.BOGUS_LABEL_VALUE: None,
  }

  # normal_data = get_df_dataset_from_name(params, data_path)
  frame_to_input = FrameToInput(params)
  frame_to_input.dataset_preprocessor.set_pipeline(
      [frame_to_input.dataset_preprocessor.check_single_image,
       frame_to_input.dataset_preprocessor.clean_misshaped,
       frame_to_input.dataset_preprocessor.select_channels,
       frame_to_input.dataset_preprocessor.normalize_by_image,
       frame_to_input.dataset_preprocessor.nan_to_num,
       frame_to_input.dataset_preprocessor.crop_at_center,
       frame_to_input.dataset_preprocessor.labels_to_real_bogus
       ])
  datasets_dict = frame_to_input.get_datasets()
  save_folder = '/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/OD_retieved_1'
  utils.check_path(save_folder)
  # saving Mayor_Test_set_(8)
  utils.save_pickle(datasets_dict[general_keys.TEST],
                    os.path.join(save_folder, 'Mayor_Test_set_(8)_dataset.pkl'))

  data_concat = np.concatenate([datasets_dict[general_keys.TRAIN].data_array,
                                datasets_dict[
                                  general_keys.VALIDATION].data_array])
  label_concat = np.concatenate([datasets_dict[general_keys.TRAIN].data_label,
                                 datasets_dict[
                                   general_keys.VALIDATION].data_label])
  normal_data = Dataset(data_array=data_concat, data_label=label_concat,
                        batch_size=None)
  print('\nCross_Match and Ashish_Bogus Counts:')
  print('Mayor_Test_set_(8):, ', np.unique(datasets_dict[general_keys.TEST].data_label,
                                 return_counts=True))
  print('Remaining: ', np.unique(normal_data.data_label,
                                 return_counts=True), '\n')

  bogus_data_name = 'bogus_juliano_franz_pancho.pkl'
  bogus_path = os.path.join(
      data_folder, 'converted_' + bogus_data_name)
  bogus_dataset = get_df_dataset_from_name(params, bogus_path)
  print('\n Alerce Bogus.shape: ', bogus_dataset.data_array.shape, '\n')
  real_data_name = 'tns_confirmed_sn.pkl'
  real_path = os.path.join(
      data_folder, 'converted_' + real_data_name)
  real_dataset = get_df_dataset_from_name(params, real_path)
  print('\n TNS SNe_(real).shape: ', real_dataset.data_array.shape, '\n')

  # 1 real - 0 bogus
  bogus_label_value = 0
  normal_data_bogus = normal_data.data_array[
    normal_data.data_label == bogus_label_value]
  # saving Bogus_Ashish_(2)
  utils.save_pickle(normal_data_bogus,
                    os.path.join(save_folder, 'Inliers_Bogus_Ashish_(2).pkl'))
  normal_data_real = normal_data.data_array[
    normal_data.data_label != bogus_label_value]

  bogus_all_samples = normal_data_bogus
  print('\n ALL Bogus.shape: ', bogus_all_samples.shape, '\n')
  # saving ALL_Bogus_(6)
  utils.save_pickle(bogus_all_samples,
                    os.path.join(save_folder, 'ALL_Bogus_(6).pkl'))


  # dont know if must mix TNS with normal cross-matched data
  normal_real_indxs = np.arange(len(normal_data_real))
  np.random.RandomState(seed=random_seed).shuffle(normal_real_indxs)
  val_size_inliers = int(
      np.round(len(normal_data_real) * val_inlier_percentage))

  # data splitting
  # TODO: Check
  normal_real_aux = normal_data_real[
    normal_real_indxs[len(bogus_all_samples):]]
  normal_real_test = normal_data_real[
    normal_real_indxs[:len(bogus_all_samples)]]
  # saving Inliers_Test_(5)
  utils.save_pickle(normal_real_test,
                    os.path.join(save_folder, 'Inliers_Test_(5).pkl'))
  normal_real_train = normal_real_aux[val_size_inliers:]
  # saving Inliers_Train_(4)
  utils.save_pickle(normal_real_train,
                    os.path.join(save_folder, 'Inliers_Train_(4).pkl'))
  normal_real_val = normal_real_aux[:val_size_inliers]
  # saving Inliers_Val_(3)
  utils.save_pickle(normal_real_val,
                    os.path.join(save_folder, 'Inliers_Val_(3).pkl'))
  print('All inliers, except mayor test: ', len(normal_data_real))
  print('Train inliers: ', len(normal_real_train))
  print('Val inliers: ', len(normal_real_val))
  print('Test inliers: ', len(normal_real_test))

  x_train = normal_real_train
  x_val = normal_real_val
  x_test = np.concatenate([normal_real_test, bogus_all_samples])
  y_test = np.concatenate(
      [np.ones(len(normal_real_test)), np.zeros(len(bogus_all_samples))])
  print('y_test: ', np.unique(y_test, return_counts=True))
  # # saving GEOT_test_Inliers_Test_U_ALL_Bogus_(5)_U_(6)
  # utils.save_pickle(Dataset(x_test, y_test, batch_size=None),
  #                   os.path.join(save_folder, 'GEOT_test_Inliers_Test_U_ALL_Bogus_(5)_U_(6)_dataset.pkl'))

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
      epochs=2  # int(np.ceil(200 / transformer.n_transforms))
  )
  met_dict = mdl.evaluate_od(
      x_train_transform, x_test_transform, y_test, 'hits-4-c', 'real',
      x_val_transform, save_hist_folder_path=mdl.specific_model_folder)

  print('\nroc_auc')
  for key in met_dict.keys():
    print(key, met_dict[key]['roc_auc'])
  print(
      "Train and evaluate %s" % utils.timer(
          start_time, time.time()))

  dirichlet_test_pred = met_dict[general_keys.DIRICHLET]['clf']
  print(np.mean(y_test == dirichlet_test_pred))

  #Saving Detected_Boguses_(7)
  bogus_save_path = os.path.join(save_folder, 'Detected_Boguses_(7).pkl')
  new_od_boguses = x_test[dirichlet_test_pred == bogus_label_value]
  print('NEW_OD_BOGUSES: ', len(new_od_boguses))
  utils.save_pickle(new_od_boguses, bogus_save_path)

  #Saving Train_Geotransform_(3)_U_(4)
  real_normal_train_val_data_path = os.path.join(save_folder,
                                                 'Train_Geotransform_(3)_U_(4).pkl')
  real_normal_train_val_data = np.concatenate([x_train, x_val])
  utils.save_pickle(real_normal_train_val_data, real_normal_train_val_data_path)

"""
"""
