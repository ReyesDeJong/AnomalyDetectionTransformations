"""
Transformer object that performs 4 Rot; 4 Translations; Flip; Kernel Ops
without their composition
"""

import abc

import numpy as np
import tensorflow as tf

from modules.geometric_transform.streaming_transformers. \
    transformation_function_builder import TransformationFunctionBuilder


class AbstractTransformer(abc.ABC):
    def __init__(self, translation_x, translation_y, rotations,
        flips, gauss, log, mixed, trivial, transform_batch_size, name):
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.rotations = rotations
        self.flips = flips
        self.gauss = gauss
        self.log = log
        self.mixed = mixed
        self.trivial = trivial
        self.name = name
        self._transform_batch_size = transform_batch_size
        self.transformation_tuples = self._get_transformation_tuples_list()
        self._create_transformation_op_list()

    @property
    def n_transforms(self):
        return len(self._transformation_ops)

    @abc.abstractmethod
    def _get_transformation_tuples_list(self):
        return

    def _create_transformation_op_list(self):
        transformation_list = []
        for is_flip, tx, ty, k_rotate, is_gauss, is_log, is_mixed, is_trivial \
            in self.transformation_tuples:
            transformation = TransformationFunctionBuilder(
                is_flip, tx, ty, k_rotate, is_gauss, is_log, is_mixed,
                is_trivial)
            transformation_list.append(transformation)
        self._transformation_ops = transformation_list

    def set_transformations_to_perform(self, transformation_list):
        # TODO: set to private
        self.transformation_tuples = transformation_list
        self._create_transformation_op_list()

    # This cannot be used, because of probles inside dataset since
    # transformation_index cannot be tensor
    def _apply_transformation_v2(self, tuple_x_transformation_index):
        x_single_image = tuple_x_transformation_index[0]
        transformation_index = tuple_x_transformation_index[1]
        x_transformed = self._transformation_ops[transformation_index](
            x_single_image[None, ...]
        )[0]
        return (x_transformed, transformation_index)

    # @tf.function
    def _apply_transformation(self, tuple_x_transformation_index):
        x_single_image = tuple_x_transformation_index[0]
        transformation_index = tuple_x_transformation_index[1]
        list_trfs = []
        for i in range(self.n_transforms):
            x_transformed = self._transformation_ops[i](
                x_single_image[None, ...])[0]
            list_trfs.append(x_transformed)
        list_trfs = tf.convert_to_tensor(list_trfs)
        x_transformed = list_trfs[transformation_index]
        return (x_transformed, transformation_index)

    # @tf.function
    def transform_batch_with_random_indexes(self, x_batch) -> (
    tf.Tensor, tf.Tensor):
        random_transformation_indexes = tf.random.uniform(
            shape=[x_batch.shape[0]], minval=0, maxval=self.n_transforms,
            dtype=tf.int32)
        x_transformed = self.transform_batch_given_indexes(
            x_batch, random_transformation_indexes)
        return x_transformed, random_transformation_indexes

    def transform_batch_with_random_categorical_indexes_for_dataset_map(
        self, x_batch, dummy_labels):
        x_transformed, transformation_indexes = \
            self.transform_batch_with_random_indexes(x_batch)
        categorical_transformation_indexes = tf.one_hot(
            transformation_indexes, depth=self.n_transforms)
        return x_transformed, categorical_transformation_indexes

    # @tf.function
    def transform_batch_given_indexes(self, x_batch,
        transformation_indexes) -> (tf.Tensor, tf.Tensor):
        x_transformed, _ = tf.map_fn(
            self._apply_transformation_v2, (x_batch, transformation_indexes),
            parallel_iterations=10)
        x_transformed_normalize = self._normalize_1_1_by_image(x_transformed)
        return x_transformed_normalize

    # @tf.function
    def apply_specific_transform(self, x_batch,
        single_transformation_index) -> (tf.Tensor, tf.Tensor):
            x_transformed = self._transformation_ops[single_transformation_index](
                x_batch)
            x_transformed_normalize = self._normalize_1_1_by_image( x_transformed)
            return x_transformed_normalize

    # TODO
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

    def _normalize_1_1_by_image(self, x_batch):
        images = x_batch
        images -= tf.reduce_min(images, axis=(1, 2), keepdims=True)
        images_max = tf.reduce_max(images, axis=(1, 2), keepdims=True)
        images_max = tf.where(tf.equal(images_max, 0), tf.ones_like(images_max),
                              images_max)
        images = images / images_max
        images = 2 * images - 1
        return images

    # TODO
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

    # TODO
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
