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

import tensorflow as tf
# from trainers.base_trainer import Trainer
from parameters import loader_keys, general_keys
# from models.transformer_od import TransformODModel
from models.transformer_od_simple_net import TransformODSimpleModel
from modules.geometric_transform.transformer_no_compositions import \
  NoCompositionTransformer
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.transform_selection.fid_modules import fid
import numpy as np
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader


# from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
# from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# data loaders
hits_params = {
  loader_keys.DATA_PATH: os.path.join(
      PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
  loader_keys.N_SAMPLES_BY_CLASS: 100000,
  loader_keys.TEST_PERCENTAGE: 0.0,
  loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
  loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #[2],#
  loader_keys.CROP_SIZE: 21,
  general_keys.RANDOM_SEED: 42,
  loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
}
hits_loader = HiTSOutlierLoader(hits_params)
hits_loader.set_pickle_loading(False)
hits_loader.set_pickle_saving(False)
data_loader = hits_loader
(x_train, y_train), (x_val, y_val), (
  x_test, y_test) = data_loader.get_outlier_detection_datasets()
print(x_train.shape)
transformer = NoCompositionTransformer()
x_train_1 = x_train[int(len(x_train)/2):]#
x_train_2 = x_train[:int(len(x_train)/2)]#[10000:20000]#
mug_orig, sigma_orig = fid.calculate_activation_statistics_from_activation_array(
    x_train_1)
# mug_orig, sigma_orig = fid.calculate_activation_statistics_from_activation_array(
#     x_train)
for transform_i in range(transformer.n_transforms):
  activations_trf_i = transformer.apply_transforms(x_train_2, [transform_i])[0]
  # activations_trf_i = transformer.apply_transforms(x_train, [transform_i])[0]
  # print(activations_trf_i.shape)
  mu_trf_i, sigma_trf_i = fid.calculate_activation_statistics_from_activation_array(
      activations_trf_i)
  # print('mu,sigma %i' % transform_i)
  fid_value = fid.calculate_frechet_distance(mug_orig, sigma_orig, mu_trf_i,
                                             sigma_trf_i)
  tuple_trf = transformer.transformation_tuples[transform_i]
  tuple_names = ['flip', 'shift', 'shift', 'rot', 'gauss_filter',
                 'laplacian_filter', 'non-transformed']
  tuple_selected_idx =  np.argwhere(np.array(tuple_trf)!=0).squeeze()
  if tuple_selected_idx.tolist() == []:
    tuple_selected_idx = -1
  print("FID non-transformed/%s %s: %f" % (tuple_names[tuple_selected_idx], tuple_trf[tuple_selected_idx], fid_value))

# ztf_params = {
#   loader_keys.DATA_PATH: os.path.join(
#       PROJECT_PATH, '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
# }
# ztf_loader = ZTFSmallOutlierLoader(ztf_params)

# data_loader = hits_loader #ztf_loader#
#
# (x_train, y_train), (x_val, y_val), (
#   x_test, y_test) = data_loader.get_outlier_detection_datasets()
#
# transformer = NoCompositionTransformer()
#
# # model = TransformODSimpleModel(data_loader=data_loader, transformer=transformer,
# #                                input_shape=x_train.shape[1:])
#
# (x_train_trf, y_train_trf), (x_val_trf, y_val_trf), (
#   x_test_trf, y_test_trf) = data_loader.get_transformed_datasets(transformer)
#
# # model.network.eval_tf(x_val_trf, tf.keras.utils.to_categorical(y_val_trf))
# # print(np.mean(np.argmax(model.network.predict(x_val_trf), axis=-1)==y_val_trf))
# # print(np.mean(model.network.get_activations(x_val_trf)))
#
# print(model.network.get_activations(x_val_trf).shape)
# # model.fit(x_train, x_val, epochs=10000, patience=0)
# # print(model.network.get_activations(x_val_trf).shape)
#
#
# # model.network.eval_tf(x_val_trf, tf.keras.utils.to_categorical(y_val_trf))
# # print(np.mean(np.argmax(model.network.predict(x_val_trf), axis=-1)==y_val_trf))
# # print(np.mean(model.network.get_activations(x_val_trf)))
#
# x_super = np.concatenate([x_train_trf, x_val_trf])
# y_super = np.concatenate([y_train_trf, y_val_trf])
#
# # activations_val = model.network.get_activations(x_super)
# activations_val = x_super.reshape(x_super.shape[0], np.prod(x_super.shape[1:]))
# activations_original_transfoms = activations_val[y_super == 0]
# mug_orig, sigma_orig = fid.calculate_activation_statistics_from_activation_array(
#     activations_original_transfoms)
#
# for transform_i in range(transformer.n_transforms):
#   activations_trf_i = activations_val[y_super == transform_i]
#   # print(activations_trf_i.shape)
#   mu_trf_i, sigma_trf_i = fid.calculate_activation_statistics_from_activation_array(
#       activations_trf_i)
#   # print('mu,sigma %i' % transform_i)
#   fid_value = fid.calculate_frechet_distance(mug_orig, sigma_orig, mu_trf_i,
#                                              sigma_trf_i)
#   tuple_trf = transformer.transformation_tuples[transform_i]
#   tuple_names = ['flip', 'shift', 'shift', 'rot', 'gauss_filter',
#                  'laplacian_filter', 'non-transformed']
#   tuple_selected_idx =  np.argwhere(np.array(tuple_trf)!=0).squeeze()
#   if tuple_selected_idx.tolist() == []:
#     tuple_selected_idx = -1
#   print("FID non-transformed/%s %s: %f" % (tuple_names[tuple_selected_idx], tuple_trf[tuple_selected_idx], fid_value))
