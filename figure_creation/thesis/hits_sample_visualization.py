"""
Visualization of HiTS sample for paper, 4 samples per class
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from parameters import loader_keys, general_keys
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
import numpy as np
import matplotlib.pyplot as plt
from modules import utils


def plot_hits_image(image, n_channels_to_plot=3, name=None, show=False,
    plot_titles=False, save_folder_name=''):
    # fill titles with blanks
    titles = ['Template', 'Science', 'Difference', 'SNR difference']
    for i in range(n_channels_to_plot):
        plt.subplot(1, n_channels_to_plot, i + 1)
        plt.imshow(image[..., i], interpolation='nearest')
        plt.axis('off')
        if plot_titles:
            plt.title(titles[i], fontdict={'fontsize': 15})
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0.1)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if name:
        save_folder_name = os.path.join(PROJECT_PATH,
                                        'figure_creation/thesis/figs',
                                        save_folder_name)
        utils.check_path(save_folder_name)
        plt.savefig(
            os.path.join(save_folder_name, '%s.svg' % name),
            format='svg', dpi=600, bbox_inches='tight', pad_inches=0,
            transparent=False)
    if show:
        plt.show()


def plot_hits_many_images(images, n_channels_to_plot=3, name=None, show=False):
    # fill titles with blanks
    titles = ['Template', 'Science', 'Difference', 'SNR difference']
    n_images = len(images)
    plt.figure(figsize=(n_images * 2, 2 * n_channels_to_plot))
    for image_i in range(n_images):
        for i in range(n_channels_to_plot):
            plt.subplot(n_images, n_channels_to_plot,
                        i + (image_i * n_channels_to_plot) + 1)
            plt.imshow(images[image_i, ..., i], interpolation='nearest')
            plt.axis('off')
            plt.title(titles[i], fontdict={'fontsize': 15})
        titles = ['', '', '', '']
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0.1)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if name:
        folder_path = os.path.join(PROJECT_PATH, 'figure_creation/thesis/figs')
        utils.check_path(folder_path)
        plt.savefig(
            os.path.join(folder_path, '%s.svg' % name),
            format='svg', dpi=600, bbox_inches='tight', pad_inches=0,
            transparent=False)
    if show:
        plt.show()


if __name__ == '__main__':
    SHOW = True
    N_SAMPLES_TO_PLOT = 5
    RANDOM_SEED = 234
    SAVE_FOLDER_NAME = 'samples_hits'
    # data loader
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 150000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: RANDOM_SEED,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    hits_loader = HiTSOutlierLoader(hits_params)
    hits_loader.set_pickle_loading(False)
    hits_loader.set_pickle_saving(False)

    # getting data
    dataset = hits_loader.get_preprocessed_unsplitted_dataset()
    dataset.shuffle_data(RANDOM_SEED)
    data_array = dataset.data_array
    data_labels = dataset.data_label
    print(dataset.data_array.shape)
    print('Data values per channel Min %s Max %s  Mean %s' % (
        np.mean(np.min(data_array, axis=(1, 2)), axis=0),
        np.mean(np.max(data_array, axis=(1, 2)), axis=0),
        np.mean(np.mean(data_array, axis=(1, 2)), axis=0)))
    print(np.unique(dataset.data_label, return_counts=True))

    # get Inliers
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(dataset.data_array[dataset.data_label == 1][i],
                        show=SHOW, name='inlier_%i' % i, plot_titles=not i,
                        save_folder_name=SAVE_FOLDER_NAME, n_channels_to_plot=data_array.shape[-1])
    # get Outliers
    for i in range(N_SAMPLES_TO_PLOT):
        plot_hits_image(dataset.data_array[dataset.data_label != 1][i],
                        show=SHOW, name='outlier_%i' % i, plot_titles=not i,
                        save_folder_name=SAVE_FOLDER_NAME, n_channels_to_plot=data_array.shape[-1])
    # plot_hits_many_images(dataset.data_array[dataset.data_label == 1][:4], show=True,
    #                name='aux')
