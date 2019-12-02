import os
import pickle as pkl
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from modules.data_set_generic import Dataset
from parameters import general_keys
from modules.data_splitter import DatasetDivider


def createCircularMask(h, w, center=None, radius=None):
  if center is None:  # use the middle of the image
    center = [int(w / 2), int(h / 2)]
  if radius is None:  # use the smallest distance between the center and image walls
    radius = min(center[0], center[1], w - center[0], h - center[1])

  Y, X = np.ogrid[:h, :w]
  dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

  mask = dist_from_center <= radius
  return mask * 1.0


def plot_n_images(dataset, name, save_path, plot_show=False, n=100,
    set_to_plot=general_keys.TRAIN):
  all_imgs = dataset[set_to_plot].data_array  #
  n_imags_available = all_imgs.shape[0]
  random_img_idxs = np.random.choice(range(n_imags_available), n)
  imgs = all_imgs[random_img_idxs, :, :, 0]
  sqrt_n = int(np.sqrt(n))
  fig, axs = plt.subplots(sqrt_n, sqrt_n, figsize=(16, 16),
                          gridspec_kw={'wspace': 0, 'hspace': 0})
  fig.suptitle(name, fontsize=40, color='white')
  axs = axs.flatten()
  for img, ax in zip(imgs, axs):
    ax.imshow(img)
    ax.axis('off')
  fig.tight_layout()
  if save_path:
    fig.savefig(os.path.join(save_path, '%s.png' % name))
  if plot_show:
    plt.show()


def generated_images_to_dataset(gen_imgs, label=1):
  dataset = Dataset(data_array=gen_imgs,
                    data_label=np.ones(gen_imgs.shape[0]) * label,
                    batch_size=50)
  data_splitter = DatasetDivider(test_size=0.12, validation_size=0.08)
  data_splitter.set_dataset_obj(dataset)
  train_dataset, test_dataset, val_dataset = \
    data_splitter.get_train_test_val_set_objs()
  datasets_dict = {
    general_keys.TRAIN: train_dataset,
    general_keys.VALIDATION: val_dataset,
    general_keys.TEST: test_dataset
  }
  return datasets_dict


def check_paths(paths):
  if not isinstance(paths, list):
    paths = [paths]
  for path in paths:
    if not os.path.exists(path):
      os.makedirs(path)


def merge_datasets_dict(datasets_dict1, datasets_dict2):
  merged_datasets_dict = {}
  for set in datasets_dict1.keys():
    data_array = np.concatenate([datasets_dict1[set].data_array,
                                 datasets_dict2[set].data_array])
    data_label = np.concatenate([datasets_dict1[set].data_label,
                                 datasets_dict2[set].data_label])
    merged_datasets_dict[set] = Dataset(data_array, data_label, batch_size=50)
  return merged_datasets_dict


def save_pickle(data, path):
  with open(path, 'wb') as handle:
    pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)


def add_text_to_beginning_of_file_path(file_path, added_text):
  """add text to beginning of file name, without modifying the base path of the original file"""
  folder_path = os.path.dirname(file_path)
  data_file_name = os.path.basename(file_path)
  converted_data_path = os.path.join(
      folder_path, '%s_%s' % (added_text, data_file_name))
  return converted_data_path


def set_soft_gpu_memory_growth():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def timer(start, end):
  hours, rem = divmod(end - start, 3600)
  minutes, seconds = divmod(rem, 60)
  return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def normalize(array, axis=-1):
  sums = np.sum(array, axis=axis)
  return array / np.expand_dims(sums, axis)
