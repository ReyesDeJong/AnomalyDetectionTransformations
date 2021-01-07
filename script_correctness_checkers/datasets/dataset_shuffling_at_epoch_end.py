"""
show bias in labels shown when shuffle d or nor
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)
from modules.data_loaders.ztf_outlier_loaderv2 import ZTFOutlierLoaderv2
from parameters import loader_keys
from modules.geometric_transform import transformations_tf
from modules import utils
import tensorflow as tf
import numpy as np


def _shuffle_dataset(x, y):
    dataset_indexes = np.arange(len(y))
    np.random.shuffle(dataset_indexes)
    x = x[dataset_indexes]
    y = y[dataset_indexes]
    return x, y


if __name__ == '__main__':
    EPOCHS = 3
    utils.set_soft_gpu_memory_growth()
    # data loaders
    # ztf
    ztf_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '..', 'datasets', 'thesis_data', 'ztfv7',
            'preprocessed_21', 'ztf_small_dict.pkl'),
    }
    ztf_loader = ZTFOutlierLoaderv2(ztf_params, 'small_ztfv7')
    outlier_loader = ztf_loader
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = outlier_loader.get_outlier_detection_datasets()
    # transformer
    transformer = transformations_tf.Transformer()  # TransTransformer()
    print('n transforms: ', transformer.n_transforms)
    x_train_transform, y_train_transform = transformer.apply_all_transforms(
        x=x_train)
    x = x_train_transform
    y = y_train_transform
    x, y = _shuffle_dataset(x, y)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x, y)).shuffle(10000).batch(1000, drop_remainder=True)
    # training loop
    for epoch in range(EPOCHS):
        print('\n epoch %i' % epoch)
        for it, (images, labels) in enumerate(train_ds):
            print('e:%i it:%i %s' % (
                epoch, it, np.unique(labels, return_counts=True)))
