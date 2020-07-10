"""
Transformer object that performs 4 Rot; 4 Translations; Flip; Kernel Ops
without their composition
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname('__file__'), '..', '..'))
sys.path.append(PROJECT_PATH)

import abc
from modules.geometric_transform. \
    transformations_tf import cnn2d_depthwise_tf, \
    makeGaussian, makeLoG, check_shape_kernel, apply_affine_transform, \
    PlusKernelTransformer
import tensorflow as tf
import itertools
import numpy as np


class AbstractTransformer(abc.ABC):
    def __init__(self, transform_batch_size=512, name='Abstract_Transformer'):
        self.name = name
        self._transform_batch_size = transform_batch_size
        self._transformation_ops = None
        self.transformation_tuples = None


    def _init_transformations(self):
        self._create_transformation_tuples_list()
        self._create_transformation_op_list()

    @property
    def n_transforms(self):
        return len(self._transformation_ops)

    @abc.abstractmethod
    def _create_transformation_op_list(self):
        return

    @abc.abstractmethod
    def _create_transformation_tuples_list(self):
        return

    def set_transformations_to_perform(self, transformation_list):
        # TODO: set to private
        self.transformation_tuples = transformation_list
        self._create_transformation_op_list()

    def _apply_transformation(self, tuple_x_transformation_index):
        x_single_image = tuple_x_transformation_index[0]
        transformation_index = tuple_x_transformation_index[1]
        x_transformed = self._transformation_ops[transformation_index](
            x_single_image[None, ...]
        )[0]
        return (x_transformed, transformation_index)

    @tf.function
    def _map_fn_wrapper(self, batch, transformation_indexes):
        return tf.map_fn(
            self._apply_transformation, (batch, transformation_indexes),
        )

    def transform_batch_given_indexes(self, x, t_inds):
        transformed_batch = []
        with tf.name_scope("transformations"):
            for i, t_ind in enumerate(t_inds):
                transformed_batch.append(self._transformation_ops[t_ind](x))
            concatenated_transformations = tf.concat(transformed_batch, axis=0)
        return tf.identity(concatenated_transformations, 'concat_transforms')

    # def apply_all_transformsv0(self, x, batch_size=None):
    #   """generate transform inds, that are the labels of each transform and
    #   its respective transformed data. It generates labels along with images"""
    #   if batch_size is not None:
    #     self._transform_batch_size = batch_size
    #   train_ds = tf.data.Dataset.from_tensor_slices((x)).batch(
    #       self._transform_batch_size)
    #   transformations_inds = np.arange(self.n_transforms)
    #   x_transform = []
    #   y_transform = []
    #   for images in train_ds:
    #     transformed_batch = transformer.transform_batch(images,
    #                                                     transformations_inds)
    #     y_transform_batch = np.repeat(
    #         transformations_inds, transformed_batch.shape[0] // self.n_transforms)
    #     x_transform.append(transformed_batch)
    #     y_transform.append(y_transform_batch)
    #   x_transform = np.concatenate(
    #       [tensor.numpy() for tensor in x_transform])
    #   y_transform = np.concatenate(y_transform)
    #   return x_transform, y_transform

    def apply_all_transforms(self, x, batch_size=None):
        """generate transform inds, that are the labels of each transform and
        its respective transformed data. It generates labels after images"""
        if self.verbose:
            if self.return_data_not_transformed:
                print('Not')
            print('Appliying all %i transforms to set of shape %s' % (
                self.n_transforms, str(x.shape)))
        transformations_inds = np.arange(self.n_transforms)
        return self.apply_transforms(x, transformations_inds, batch_size)

    def _normalize_1_1_by_image(self, image_array):
        images = image_array
        images -= np.nanmin(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        images_max = np.nanmax(images, axis=(1, 2))[
                     :, np.newaxis, np.newaxis, :]
        images_max[images_max == 0] = 1
        images = images / images_max
        images = 2 * images - 1
        return images

    def apply_transforms(self, x, transformations_inds, batch_size=None):
        """generate transform inds, that are the labels of each transform and
        its respective transformed data. It generates labels after images"""
        if batch_size is not None:
            self._transform_batch_size = batch_size
        if self.return_data_not_transformed:
            self.original_x_len = int(len(x) / self.n_transforms)
            y_transformed = self._get_y_transform(self.original_x_len,
                                                  transformations_inds)
            return x, y_transformed

        train_ds = tf.data.Dataset.from_tensor_slices((x)).batch(
            self._transform_batch_size)

        # Todo: check which case is faste, if same, keep second way, it uses less memory
        # if x.shape[1] != 63:  # or self.n_transforms>90:
        #  x_transform = []
        #  for images in train_ds:
        #    transformed_batch = self.transform_batch(images, transformations_inds)
        #    x_transform.append(transformed_batch)
        #  x_transform = np.concatenate(
        #      [tensor.numpy() for tensor in x_transform])
        # else:
        x_transform = np.empty(
            (x.shape[0] * len(transformations_inds), x.shape[1], x.shape[2],
             x.shape[3]),
            dtype=np.float32)
        i = 0
        for images in train_ds:
            transformed_batch = self.transform_batch(images,
                                                     transformations_inds)
            x_transform[
            i:i + self._transform_batch_size * len(transformations_inds)] = \
                transformed_batch.numpy()
            i += self._transform_batch_size * len(transformations_inds)
        self.original_x_len = len(x)
        y_transform = self._get_y_transform(self.original_x_len,
                                            transformations_inds)
        del train_ds
        x_transform = self._normalize_1_1_by_image(x_transform)
        return x_transform, y_transform

    def _get_y_transform(self, len_x, transformations_inds):
        y_transform_fixed_batch_size = np.repeat(transformations_inds,
                                                 self._transform_batch_size)
        y_transform_fixed_batch_size = np.tile(y_transform_fixed_batch_size,
                                               len_x // self._transform_batch_size)
        y_transform_leftover_batch_size = np.repeat(
            transformations_inds, len_x % self._transform_batch_size)
        y_transform = np.concatenate(
            [y_transform_fixed_batch_size, y_transform_leftover_batch_size])
        return y_transform


class RankingTransformation(object):
    def __init__(self, flip, tx, ty, k_90_rotate, gauss, log, mixed, trivial):
        self.flip = flip
        self.tx = tx
        self.ty = ty
        self.k_90_rotate = k_90_rotate
        self.gauss = gauss
        self.log = log
        self.gauss_kernel = makeGaussian(5, 1).astype(np.float32)
        self.log_kernel = makeLoG(5, 0.5).astype(np.float32)
        self.mixed = mixed
        self.trivial = trivial

    def __call__(self, x):
        res_x = x
        if self.gauss:
            res_x = cnn2d_depthwise_tf(
                res_x, check_shape_kernel(self.gauss_kernel, res_x))
        if self.log:
            res_x = cnn2d_depthwise_tf(
                res_x, check_shape_kernel(self.log_kernel, res_x))
        if self.flip:
            with tf.name_scope("flip"):
                res_x = tf.image.flip_left_right(res_x)
        if self.tx != 0 or self.ty != 0:
            res_x = apply_affine_transform(res_x, self.tx, self.ty)
        if self.k_90_rotate != 0:
            with tf.name_scope("rotation"):
                res_x = tf.image.rot90(res_x, k=self.k_90_rotate)
        if self.mixed:
            # print('mixed')
            # print(x.shape)
            # print(tf.reduce_mean(x[0,:,:,0]))
            flatten_img_x = tf.reshape(x, [x.shape[0], x.shape[1] * x.shape[2],
                                           x.shape[3]])
            # print(flatten_img_x.shape)
            # print(tf.reduce_mean(flatten_img_x[0, :, 0]))
            perm_x = tf.transpose(flatten_img_x, [1, 0, 2])
            # print(perm_x.shape)
            # print(tf.reduce_mean(perm_x[:, 0, 0]))
            shufled_x = tf.random.shuffle(perm_x)
            # print(shufled_x.shape)
            # print(tf.reduce_mean(shufled_x[:, 0, 0]))
            perm2_x = tf.transpose(shufled_x, [1, 0, 2])
            # print(perm2_x.shape)
            # print(tf.reduce_mean(perm2_x[0, :, 0]))
            reshaped_x = tf.reshape(perm2_x, [x.shape[0], x.shape[1],
                                              x.shape[2], x.shape[3]])
            # print(reshaped_x.shape)
            # print(tf.reduce_mean(reshaped_x[0, :, :, 0]))
            res_x = reshaped_x
            # print(res_x.shape)
            # print(tf.reduce_mean(res_x[0, :, :, 0]))
            # OLD
            # reshaped_x = tf.reshape(shufled_x, [perm_x.shape[0], perm_x.shape[1],
            #                               perm_x.shape[2], perm_x.shape[3]])
            # res_x = tf.transpose(reshaped_x, [2, 0, 1, 3])
        if self.trivial:
            # print('trivial')
            res_x = x * 0 + tf.random.normal(x.shape)
        return res_x


class RankingTransformer(AbstractTransformer):
    def __init__(self, translation_x=8, translation_y=8, rotations=True,
        flips=True, gauss=True, log=True, mixed=1, trivial=1,
        transform_batch_size=512, name='Ranking_Transformer'):
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.rotations = rotations
        self.flips = flips
        self.gauss = gauss
        self.log = log
        self.mixed = mixed
        self.trivial = trivial
        super().__init__(transform_batch_size, name)

    def _create_transformation_tuples_list(self):
        self.transformation_tuples = (
            (0, 0, 0, 0, 0, 0, 0, 0), (1 * self.flips, 0, 0, 0, 0, 0, 0, 0),
            (0, self.translation_x, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, self.rotations * 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 1 * self.gauss, 0, 0, 0),
            (0, 0, 0, 0, 0, 1 * self.log, 0, 0),
            (0, 0, 0, 0, 0, 0, self.mixed, 0),
            (0, 0, 0, 0, 0, 0, 0, self.trivial),
        )
        # if some of the parameters is st to zero, avoid transformation redundance,
        # because original would appear more than once
        if self.translation_y * self.translation_x * self.rotations * \
            self.flips * self.gauss * self.log * self.trivial * self.mixed == 0:
            self.transformation_tuples = tuple(
                np.unique(self.transformation_tuples, axis=0))

    def _create_transformation_op_list(self):
        transformation_list = []
        for is_flip, tx, ty, k_rotate, is_gauss, is_log, is_mixed, is_trivial, in \
            self.transformation_tuples:
            transformation = RankingTransformation(
                is_flip, tx, ty, k_rotate, is_gauss, is_log, is_mixed,
                is_trivial)
            transformation_list.append(transformation)

        self._transformation_ops = transformation_list


# TODO: see if can do some refactoring here
class PlusKernelShuffleNoiseTransformer(PlusKernelTransformer):
    def __init__(self, translation_x=8, translation_y=8, rotations=True,
        flips=True, gauss=True, log=True, mixed=1, trivial=1,
        transform_batch_size=512,
        name='PlusKernelShuffleNoise_Transformer'):
        self.mixed = mixed
        self.trivial = trivial
        super().__init__(
            translation_x, translation_y, rotations,
            flips, gauss, log, transform_batch_size, name)

    def _create_transformation_tuples_list(self):
        self.transformation_tuples = list(itertools.product(
            self.iterable_flips,
            self.iterable_tx,
            self.iterable_ty,
            self.iterable_rot,
            [0],
            [0],
            [0],
            [0]))
        self.transformation_tuples += list(itertools.product(
            [0],
            self.iterable_tx,
            self.iterable_ty,
            [0],
            [1],
            [0],
            [0],
            [0]))
        self.transformation_tuples += list(itertools.product(
            [0],
            self.iterable_tx,
            self.iterable_ty,
            [0],
            [0],
            [1],
            [0],
            [0]))
        self.transformation_tuples += list(itertools.product(
            [0],
            self.iterable_tx,
            self.iterable_ty,
            [0],
            [1],
            [1],
            [0],
            [0]))
        self.transformation_tuples += ((0, 0, 0, 0, 0, 0, self.mixed, 0),
                                       (0, 0, 0, 0, 0, 0, 0, self.trivial))

    def _create_transformation_op_list(self):
        transformation_list = []
        for is_flip, tx, ty, k_rotate, is_gauss, is_log, is_mixed, is_trivial, in \
            self.transformation_tuples:
            transformation = RankingTransformation(
                is_flip, tx, ty, k_rotate, is_gauss, is_log, is_mixed,
                is_trivial)
            transformation_list.append(transformation)

        self._transformation_ops = transformation_list


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

    transformer = RankingTransformer()
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


if __name__ == "__main__":
    test_visualize_transforms()
