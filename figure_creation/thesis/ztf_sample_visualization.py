"""
Visualization of HiTS sample for paper, 4 samples per class
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

import matplotlib.pyplot as plt
from parameters import param_keys
from modules.data_loaders.frame_to_input import FrameToInput
import numpy as np
from modules import utils


def plot_ztf_image(image, n_channels_to_plot=3, name=None, show=False,
    plot_titles=False, save_folder_name=''):
    # fill titles with blanks
    titles = ['Template', 'Science', 'Difference']
    for i in range(n_channels_to_plot):
        j = i
        if i == 0:
            j = 1
        if i == 1:
            j = 0
        plt.subplot(1, n_channels_to_plot, i + 1)
        plt.imshow(image[..., j], interpolation='nearest', cmap='gray')
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


if __name__ == '__main__':
    SHOW = True
    N_SAMPLES_TO_PLOT = 5
    RANDOM_SEED = 234
    SAVE_FOLDER_NAME = 'samples_ztf'

    # data loader
    data_name = 'training_set_Aug-07-2020.pkl'
    data_folder = "/home/ereyes/Projects/Thesis/stamp_classifier_updater/data/"
    data_path = os.path.join(data_folder, data_name)
    n_classes = 5
    params = {
        # param_keys.DATA_PATH_TRAIN: '/home/ereyes/Projects/Thesis/'
        #                                     'datasets/ALeRCE_data/ztf_v5/'
        #                                     'converted_data.pkl',
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.BATCH_SIZE: None,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.NANS_TO: 0,
        param_keys.CROP_SIZE: 63,
        param_keys.TEST_SIZE: n_classes * 300,
        param_keys.VAL_SIZE: n_classes * 100,
        param_keys.VALIDATION_RANDOM_SEED: RANDOM_SEED,
        param_keys.CONVERTED_DATA_SAVEPATH: '/home/ereyes/Projects/Thesis/'
                                            'datasets/ALeRCE_data/ztf_v5/'
                                            'converted_data.pkl',
        param_keys.BOGUS_LABEL_VALUE: None,
    }
    frame_to_input = FrameToInput(params)
    frame_to_input.dataset_preprocessor.set_pipeline(
        [frame_to_input.dataset_preprocessor.check_single_image,
         frame_to_input.dataset_preprocessor.clean_misshaped,
         frame_to_input.dataset_preprocessor.select_channels,
         frame_to_input.dataset_preprocessor.crop_at_center,
         frame_to_input.dataset_preprocessor.normalize_by_image,
         frame_to_input.dataset_preprocessor.nan_to_num,
         ])

    # getting data
    dataset = frame_to_input.get_single_dataset()
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
        label_value = int(i % 4)
        plot_ztf_image(dataset.data_array[dataset.data_label == label_value][i],
                       show=SHOW, name='inlier_%i_class_%i' % (i, label_value),
                       plot_titles=not i,
                       save_folder_name=SAVE_FOLDER_NAME,
                       n_channels_to_plot=data_array.shape[-1])
    # get Outliers
    for i in range(N_SAMPLES_TO_PLOT):
        plot_ztf_image(dataset.data_array[dataset.data_label == 4][i],
                       show=SHOW, name='outlier_%i' % i, plot_titles=not i,
                       save_folder_name=SAVE_FOLDER_NAME,
                       n_channels_to_plot=data_array.shape[-1])

    # plot_hits_many_images(dataset.data_array[dataset.data_label == 1][:4], show=True,
    #                name='aux')
