import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from models.gans.base_gan import BaseGAN
from modules.data_loaders.hits_loader import HiTSLoader
from modules.data_set_generic import Dataset
from parameters import param_keys, general_keys
from modules.data_splitter import DatasetDivider
import matplotlib as mpl
from scipy.special import gammaincinv


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


def check_path(path):
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