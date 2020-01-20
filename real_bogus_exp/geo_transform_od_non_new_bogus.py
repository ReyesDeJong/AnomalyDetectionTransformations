import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from parameters import param_keys
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

  params = {
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.BATCH_SIZE: None,
    param_keys.N_INPUT_CHANNELS: 3,
    param_keys.CHANNELS_TO_USE: [0, 1, 2],
    param_keys.TEST_SIZE: None,
    param_keys.VAL_SIZE: None,
    param_keys.NANS_TO: 0,
    param_keys.CROP_SIZE: 21,
    param_keys.VALIDATION_RANDOM_SEED: 42,
    param_keys.CONVERTED_DATA_SAVEPATH: None,
    param_keys.BOGUS_LABEL_VALUE: None,
  }

  normal_data = get_df_dataset_from_name(params, data_path)
  bogus_data_name = 'bogus_juliano_franz_pancho.pkl'
  bogus_path = os.path.join(
      data_folder, 'converted_' + bogus_data_name)
  bogus_dataset = get_df_dataset_from_name(params, bogus_path)
  print('bogus.shape: ', bogus_dataset.data_array.shape)
  real_data_name = 'tns_confirmed_sn.pkl'
  real_path = os.path.join(
      data_folder, 'converted_' + real_data_name)
  real_dataset = get_df_dataset_from_name(params, real_path)

  # 1 real - 0 bogus
  bogus_label_value = 0
  normal_data_bogus = normal_data.data_array[
    normal_data.data_label == bogus_label_value]
  normal_data_real = normal_data.data_array[
    normal_data.data_label != bogus_label_value]

  # bogus_all_samples = np.concatenate(
  #     [normal_data_bogus, bogus_dataset.data_array])
  bogus_all_samples = normal_data_bogus

  # dont know if must mix TNS with normal cross-matched data
  normal_real_indxs = np.arange(len(normal_data_real))
  np.random.RandomState(seed=random_seed).shuffle(normal_real_indxs)
  val_size_inliers = int(
    np.round(len(normal_data_real) * val_inlier_percentage))

  #data splitting
  #TODO: Check
  normal_real_aux = normal_data_real[
    normal_real_indxs[len(bogus_all_samples):]]
  normal_real_test = normal_data_real[
    normal_real_indxs[:len(bogus_all_samples)]]
  normal_real_train = normal_real_aux[val_size_inliers:]
  normal_real_val = normal_real_aux[:val_size_inliers]
  print(len(normal_data_real))
  print(len(normal_real_train))
  print(len(normal_real_val))
  print(len(normal_real_test))


  x_train = normal_real_train
  x_val = normal_real_val
  x_test = np.concatenate([normal_real_test, bogus_all_samples])
  y_test = np.concatenate(
      [np.ones(len(normal_real_test)), np.zeros(len(bogus_all_samples))])
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
      epochs=2  # int(np.ceil(200 / transformer.n_transforms))
  )
  met_dict = mdl.evaluate_od(
      x_train_transform,  x_test_transform, y_test, 'hits-4-c', 'real',
      x_val_transform, save_hist_folder_path=mdl.specific_model_folder)

  print('\nroc_auc')
  for key in met_dict.keys():
    print(key, met_dict[key]['roc_auc'])
  print(
      "Train and evaluate %s" % utils.timer(
          start_time, time.time()))
