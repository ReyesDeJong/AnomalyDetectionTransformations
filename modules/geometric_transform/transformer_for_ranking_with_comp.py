"""
Transformer object that performs 4 Rot; 4 Translations; Flip; Kernel Ops
without their composition
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname('__file__'), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform. \
    transformations_tf import PlusKernelTransformer
import tensorflow as tf
import itertools
import numpy as np
from modules.geometric_transform.transformer_for_ranking import \
    RankingTransformer


class RankingSingleCompositionTransformer(RankingTransformer):
    def __init__(self, translation_x=8, translation_y=8, rotations=True,
        flips=True, gauss=True, log=True, mixed=1, noise=1, trivial=1,
        transform_batch_size=512, name='Ranking_Single_Composition_Transformer'):
        super().__init__(translation_x, translation_y, rotations, flips, gauss,
                         log, mixed, noise, trivial, transform_batch_size, name)

    def _create_transformation_tuples_list(self):
        self.transformation_tuples = (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (1 * self.flips, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, self.translation_x, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, self.rotations * 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 1 * self.gauss, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1 * self.log, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, self.mixed, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, self.noise, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, self.trivial),
            (1 * self.flips, self.translation_x, 0, 0, 0, 0, 0, 0, 0),
        )
        # if some of the parameters is st to zero, avoid transformation redundance,
        # because original would appear more than once
        if self.translation_y * self.translation_x * self.rotations * \
            self.flips * self.gauss * self.log * self.noise * self.mixed * \
            self.trivial == 0:
            self.transformation_tuples = tuple(
                np.unique(self.transformation_tuples, axis=0))

def test_visualize_transforms():
    import imageio
    import os
    import matplotlib.pyplot as plt

    im_path = os.path.join(PROJECT_PATH, 'extra_files', 'dragon.png')

    im = imageio.imread(im_path)

    im = im[np.newaxis, :150, :150, :3]
    im = im / np.max(im)
    print(im.shape)
    plt.imshow(im[0])
    plt.show()

    transformer = RankingSingleCompositionTransformer()
    print(transformer.n_transforms)
    transformations_inds = np.arange(transformer.n_transforms)

    transformed_batch = transformer.transform_batch(
        tf.convert_to_tensor(im, dtype=tf.float32),
        transformations_inds)

    print(transformed_batch.shape)

    for i in range(transformer.n_transforms):
        transform_indx = i
        plt.imshow(transformed_batch[transform_indx])
        plt.title(str(transformer.transformation_tuples[i]))
        plt.axis('off')
        plt.show()

def test_visualize_transforms_hits():
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from parameters import loader_keys, general_keys
    import imageio
    import os
    import matplotlib.pyplot as plt

    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 10000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    data_loader = HiTSOutlierLoader(hits_params)
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = data_loader.get_outlier_detection_datasets()

    im = x_test[0, :, :, -1][None, ..., None]
    print(im.shape)
    plt.imshow(im[0, ..., 0])
    plt.show()

    transformer = RankingSingleCompositionTransformer()
    print(transformer.n_transforms)
    transformations_inds = np.arange(transformer.n_transforms)

    transformed_batch = transformer.transform_batch(
        tf.convert_to_tensor(im, dtype=tf.float32),
        transformations_inds)

    print(transformed_batch.shape)

    for i in range(transformer.n_transforms):
        transform_indx = i
        plt.imshow(transformed_batch[transform_indx, ..., 0])
        plt.title(str(transformer.transformation_tuples[i]))
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # test_visualize_transforms()
    test_visualize_transforms_hits()
