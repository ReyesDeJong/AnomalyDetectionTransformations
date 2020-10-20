import os
import pprint
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)
# import cv2
from modules.data_loaders.hits_loader import HiTSLoader
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import param_keys, loader_keys, general_keys
import numpy as np
from tensorflow.keras.backend import cast_to_floatx
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100, cifar10
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from modules.data_loaders.frame_to_input import FrameToInput


# def resize_and_crop_image(input_file, output_side_length, greyscale=False):
#   img = cv2.imread(input_file)
#   img = cv2.cvtColor(img,
#                      cv2.COLOR_BGR2RGB if not greyscale else cv2.COLOR_BGR2GRAY)
#   height, width = img.shape[:2]
#   new_height = output_side_length
#   new_width = output_side_length
#   if height > width:
#     new_height = int(output_side_length * height / width)
#   else:
#     new_width = int(output_side_length * width / height)
#   resized_img = cv2.resize(img, (new_width, new_height),
#                            interpolation=cv2.INTER_AREA)
#   height_offset = (new_height - output_side_length) // 2
#   width_offset = (new_width - output_side_length) // 2
#   cropped_img = resized_img[height_offset:height_offset + output_side_length,
#                 width_offset:width_offset + output_side_length]
#   assert cropped_img.shape[:2] == (output_side_length, output_side_length)
#   return cropped_img
#

def normalize_minus1_1(data):
  return 2 * (data / 255.) - 1


def normalize_hits_minus1_1(data):
  images = data
  images -= np.nanmin(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
  images = images / np.nanmax(images, axis=(1, 2))[
                    :, np.newaxis, np.newaxis, :]
  images = 2 * images - 1
  return images


def get_channels_axis():
  import tensorflow.keras as keras
  idf = keras.backend.image_data_format()
  if idf == 'channels_first':
    return 1
  assert idf == 'channels_last'
  return 3


def load_fashion_mnist():
  (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
  X_train = normalize_minus1_1(
      cast_to_floatx(np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')))
  X_train = np.expand_dims(X_train, axis=get_channels_axis())
  X_test = normalize_minus1_1(
      cast_to_floatx(np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')))
  X_test = np.expand_dims(X_test, axis=get_channels_axis())
  return (X_train, y_train), (X_test, y_test)


def load_mnist():
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  X_train = normalize_minus1_1(
      cast_to_floatx(np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')))
  X_train = np.expand_dims(X_train, axis=get_channels_axis())
  X_test = normalize_minus1_1(
      cast_to_floatx(np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')))
  X_test = np.expand_dims(X_test, axis=get_channels_axis())
  return (X_train, y_train), (X_test, y_test)


def load_cifar10():
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()
  X_train = normalize_minus1_1(cast_to_floatx(X_train))
  X_test = normalize_minus1_1(cast_to_floatx(X_test))
  return (X_train, y_train), (X_test, y_test)


def load_cifar100(label_mode='coarse'):
  (X_train, y_train), (X_test, y_test) = cifar100.load_data(
      label_mode=label_mode)
  X_train = normalize_minus1_1(cast_to_floatx(X_train))
  X_test = normalize_minus1_1(cast_to_floatx(X_test))
  return (X_train, y_train), (X_test, y_test)


def save_roc_pr_curve_data(scores, labels, file_path=None):
  scores = scores.flatten()
  labels = labels.flatten()

  scores_pos = scores[labels == 1]
  scores_neg = scores[labels != 1]

  truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
  preds = np.concatenate((scores_neg, scores_pos))
  fpr, tpr, roc_thresholds = roc_curve(truth, preds)
  roc_auc = auc(fpr, tpr)

  # pr curve where "normal" is the positive class
  precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(
      truth, preds)
  pr_auc_norm = auc(recall_norm, precision_norm)

  # pr curve where "anomaly" is the positive class
  precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(
      truth, -preds, pos_label=0)
  pr_auc_anom = auc(recall_anom, precision_anom)

  if file_path is not None:
    np.savez_compressed(file_path,
                        preds=preds, truth=truth,
                        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds,
                        roc_auc=roc_auc,
                        precision_norm=precision_norm, recall_norm=recall_norm,
                        pr_thresholds_norm=pr_thresholds_norm,
                        pr_auc_norm=pr_auc_norm,
                        precision_anom=precision_anom, recall_anom=recall_anom,
                        pr_thresholds_anom=pr_thresholds_anom,
                        pr_auc_anom=pr_auc_anom)
  else:
    pprint.pprint({'fpr': fpr, 'tpr': tpr, 'roc_thresholds': roc_thresholds,
                   'roc_auc': roc_auc,
                   'precision_norm': precision_norm, 'recall_norm': recall_norm,
                   'pr_thresholds_norm': pr_thresholds_norm,
                   'pr_auc_norm': pr_auc_norm,
                   'precision_anom': precision_anom, 'recall_anom': recall_anom,
                   'pr_thresholds_anom': pr_thresholds_anom,
                   'pr_auc_anom': pr_auc_anom})


# def create_cats_vs_dogs_npz(cats_vs_dogs_path='./'):
#   labels = ['cat', 'dog']
#   label_to_y_dict = {l: i for i, l in enumerate(labels)}
#
#   def _load_from_dir(dir_name):
#     glob_path = os.path.join(cats_vs_dogs_path, dir_name, '*.*.jpg')
#     imgs_paths = glob(glob_path)
#     images = [resize_and_crop_image(p, 64) for p in imgs_paths]
#     x = np.stack(images)
#     y = [label_to_y_dict[os.path.split(p)[-1][:3]] for p in imgs_paths]
#     y = np.array(y)
#     return x, y
#
#   x_train, y_train = _load_from_dir('train')
#   x_test, y_test = _load_from_dir('test')
#
#   np.savez_compressed(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'),
#                       x_train=x_train, y_train=y_train,
#                       x_test=x_test, y_test=y_test)
#
#
# def load_cats_vs_dogs(cats_vs_dogs_path='./'):
#   npz_file = np.load(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'))
#   x_train = normalize_minus1_1(cast_to_floatx(npz_file['x_train']))
#   y_train = npz_file['y_train']
#   x_test = normalize_minus1_1(cast_to_floatx(npz_file['x_test']))
#   y_test = npz_file['y_test']
#
#   return (x_train, y_train), (x_test, y_test)


def load_hits_padded(n_samples_by_class=12500 * 2):
  data_path = os.path.join(PROJECT_PATH, '..', 'datasets',
                           'HiTS2013_300k_samples.pkl')
  params = {
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.BATCH_SIZE: 50
  }
  hits_loader = HiTSLoader(params, label_value=-1,
                           first_n_samples_by_class=n_samples_by_class)

  (X_train, y_train), (X_test, y_test) = hits_loader.load_data()

  X_train = normalize_hits_minus1_1(
      cast_to_floatx(
          np.pad(X_train, ((0, 0), (6, 5), (6, 5), (0, 0)), 'constant')))
  X_test = normalize_hits_minus1_1(
      cast_to_floatx(
          np.pad(X_test, ((0, 0), (6, 5), (6, 5), (0, 0)), 'constant')))
  return (X_train, y_train), (X_test, y_test)


def load_hits(n_samples_by_class=10000, test_size=0.20, val_size=0.10,
    return_val=False, channels_to_get=[0, 1, 2, 3]):  # [2]):  #
  data_path = os.path.join(PROJECT_PATH, '..', 'datasets',
                           'HiTS2013_300k_samples.pkl')
  params = {
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.BATCH_SIZE: 50
  }
  hits_loader = HiTSLoader(params, label_value=-1,
                           first_n_samples_by_class=n_samples_by_class,
                           test_size=test_size, validation_size=val_size,
                           channels_to_get=channels_to_get)

  (X_train, y_train), (X_val, y_val), (X_test, y_test) = hits_loader.load_data()

  X_train = normalize_hits_minus1_1(cast_to_floatx(X_train))
  X_val = normalize_hits_minus1_1(cast_to_floatx(X_val))
  X_test = normalize_hits_minus1_1(cast_to_floatx(X_test))

  if return_val:
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
  return (X_train, y_train), (X_test, y_test)

def load_hits4c(n_samples_by_class=10000, test_size=0.20, val_size=0.10,
    return_val=False, channels_to_get=[0, 1, 2, 3]):  # [2]):  #
  data_path = os.path.join(PROJECT_PATH, '..', 'datasets',
                           'HiTS2013_300k_samples.pkl')
  params = {
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.BATCH_SIZE: 50
  }
  hits_loader = HiTSLoader(params, label_value=-1,
                           first_n_samples_by_class=n_samples_by_class,
                           test_size=test_size, validation_size=val_size,
                           channels_to_get=channels_to_get)

  (X_train, y_train), (X_val, y_val), (X_test, y_test) = hits_loader.load_data()

  X_train = normalize_hits_minus1_1(cast_to_floatx(X_train))
  X_val = normalize_hits_minus1_1(cast_to_floatx(X_val))
  X_test = normalize_hits_minus1_1(cast_to_floatx(X_test))

  if return_val:
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
  return (X_train, y_train), (X_test, y_test)

def load_hits1c(n_samples_by_class=10000, test_size=0.20, val_size=0.10,
    return_val=False, channels_to_get=[2]):  #
  data_path = os.path.join(PROJECT_PATH, '..', 'datasets',
                           'HiTS2013_300k_samples.pkl')
  params = {
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.BATCH_SIZE: 50
  }
  hits_loader = HiTSLoader(params, label_value=-1,
                           first_n_samples_by_class=n_samples_by_class,
                           test_size=test_size, validation_size=val_size,
                           channels_to_get=channels_to_get)

  (X_train, y_train), (X_val, y_val), (X_test, y_test) = hits_loader.load_data()

  X_train = normalize_hits_minus1_1(cast_to_floatx(X_train))
  X_val = normalize_hits_minus1_1(cast_to_floatx(X_val))
  X_test = normalize_hits_minus1_1(cast_to_floatx(X_test))

  if return_val:
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
  return (X_train, y_train), (X_test, y_test)


def load_ztf_real_bog(val_percentage_of_inliers=0.10,
    return_val=False, channels_to_get=[0, 1, 2],
    data_file_name='ztf_v1_bogus_added.pkl', crop_size=21):
  """Load and already separated inlier-outlier as real-bogus dataset, where label 1 is real."""
  folder_path = os.path.join(PROJECT_PATH, '..', 'datasets')
  data_path = os.path.join(folder_path, data_file_name)
  # check if preprocessed data (converted) already exists
  converted_data_path = os.path.join(folder_path, 'converted%s_%s' % (
    str(crop_size), data_file_name))
  if os.path.exists(converted_data_path):
    data_path = converted_data_path

  # params for Frame input ztf loader and preprocessing
  params = {
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.BATCH_SIZE: 0,
    param_keys.CHANNELS_TO_USE: channels_to_get,
    param_keys.TEST_SIZE: 0,  # not used
    param_keys.VAL_SIZE: 0,  # not used
    param_keys.NANS_TO: 0,
    param_keys.CROP_SIZE: crop_size,
    param_keys.CONVERTED_DATA_SAVEPATH: converted_data_path
  }
  # instantiate laoder, set preprocessor, load dataset
  data_loader = FrameToInput(params)
  data_loader.dataset_preprocessor.set_pipeline(
      [data_loader.dataset_preprocessor.image_check_single_image,
       data_loader.dataset_preprocessor.image_clean_misshaped,
       data_loader.dataset_preprocessor.image_select_channels,
       data_loader.dataset_preprocessor.image_crop_at_center,
       data_loader.dataset_preprocessor.image_normalize_by_image_1_1,
       data_loader.dataset_preprocessor.image_nan_to_num
       ])
  dataset = data_loader.get_single_dataset()
  # labels from 5 classes to 0-1 as bogus-real
  bogus_class_indx = 4
  new_labels = (dataset.data_label.flatten() != bogus_class_indx) * 1.0
  # print(np.unique(new_labels, return_counts=True))
  inlier_task = 1

  # separate data into train-val-test
  outlier_indexes = np.where(new_labels != inlier_task)[0]
  inlier_indexes = np.where(new_labels == inlier_task)[0]
  if crop_size is None:
    inlier_indexes = inlier_indexes[:10000]
  # real == inliers
  val_size_inliers = int(
    np.round(len(inlier_indexes) * val_percentage_of_inliers))
  np.random.shuffle(inlier_indexes)
  # train-val indexes inlier indexes
  train_inlier_idxs = inlier_indexes[val_size_inliers:]
  val_inlier_idxs = inlier_indexes[:val_size_inliers]
  # train-test inlier indexes
  n_outliers = np.sum(new_labels != inlier_task)
  train_inlier_idxs = train_inlier_idxs[n_outliers:]
  test_inlier_idxs = train_inlier_idxs[:n_outliers]

  X_train, y_train = dataset.data_array[train_inlier_idxs], new_labels[
    train_inlier_idxs]
  X_val, y_val = dataset.data_array[val_inlier_idxs], new_labels[
    val_inlier_idxs]
  X_test, y_test = np.concatenate(
      [dataset.data_array[test_inlier_idxs],
       dataset.data_array[outlier_indexes]]), np.concatenate(
      [new_labels[test_inlier_idxs], new_labels[outlier_indexes]])
  # print('train: ', np.unique(y_train, return_counts=True))
  # print('val: ', np.unique(y_val, return_counts=True))
  # print('test: ', np.unique(y_test, return_counts=True))
  if return_val:
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
  return (X_train, y_train), (X_test, y_test)


def load_hits4c_outlier_loader(return_val=False):
  # data loaders
  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [0, 1, 2, 3],  # [2],#
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  hits_loader = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = hits_loader.get_outlier_detection_datasets()

  if return_val:
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
  return (x_train, y_train), (x_test, y_test)

def load_ztf_small(return_val=False):
  ztf_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
  }
  ztf_loader = ZTFSmallOutlierLoader(ztf_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = ztf_loader.get_outlier_detection_datasets()

  if return_val:
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
  return (x_train, y_train), (x_test, y_test)

def get_class_name_from_index(index, dataset_name):
  ind_to_name = {
    'cifar10': (
      'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
      'ship', 'truck'),
    'cifar100': ('aquatic mammals', 'fish', 'flowers', 'food containers',
                 'fruit and vegetables',
                 'household electrical devices', 'household furniture',
                 'insects', 'large carnivores',
                 'large man-made outdoor things',
                 'large natural outdoor scenes',
                 'large omnivores and herbivores',
                 'medium-sized mammals', 'non-insect invertebrates', 'people',
                 'reptiles', 'small mammals', 'trees',
                 'vehicles 1', 'vehicles 2'),
    'fashion-mnist': ('t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                      'sandal', 'shirt', 'sneaker', 'bag', 'ankle-boot'),
    'mnist': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
    'cats-vs-dogs': ('cat', 'dog'),
    'hits': ('bogus', 'real'),
    'hits-4-c': ('bogus', 'real'),
    'hits-4-c-od': ('bogus', 'real'),
    'hits-1-c': ('bogus', 'real'),
    'hits-padded': ('bogus', 'real'),
    'ztf-small-real-bog': ('bogus', 'real'),
    'ztf-real-bog': ('bogus', 'real'),
    'ztf-real-bog-v0': ('bogus', 'real'),
    'ztf-real-bog-v1': ('bogus', 'real'),
    'ztf-real-bog-v1-no-crop': ('bogus', 'real'),
  }

  return ind_to_name[dataset_name][index]


if __name__ == '__main__':
  (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_ztf_real_bog(
      return_val=True)

  print('train: ', np.unique(y_train, return_counts=True))
  print('val: ', np.unique(y_val, return_counts=True))
  print('test: ', np.unique(y_test, return_counts=True))
  import matplotlib.pyplot as plt

  plt.imshow(x_test[-1][..., -1])
  plt.show()
