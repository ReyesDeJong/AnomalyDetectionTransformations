"""
Badly done, shuffle across batch dimensions
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname('__file__'), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform. \
  transformations_tf import AbstractTransformer, cnn2d_depthwise_tf, \
  makeGaussian, makeLoG, check_shape_kernel, apply_affine_transform
import tensorflow as tf

import numpy as np


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
    if self.mixed != 0:
      perm_x = tf.transpose(x, [1, 2, 0, 3])
      to_shuffle_x = tf.reshape(x, [perm_x.shape[0] * perm_x.shape[1],
                                    perm_x.shape[2], perm_x.shape[3]])
      shufled_x = tf.random.shuffle(to_shuffle_x)
      reshaped_x = tf.reshape(shufled_x, [perm_x.shape[0], perm_x.shape[1],
                                    perm_x.shape[2], perm_x.shape[3]])
      res_x = tf.transpose(reshaped_x, [2, 0, 1, 3])
    if self.trivial != 0:
      res_x = x * 0 + tf.random.normal(x.shape)
    return res_x


class RankingTransformerOld(AbstractTransformer):
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
      (0, 0, 0, 0, 1 * self.gauss, 0, 0, 0),
      (0, 0, 0, 0, 0, 1 * self.log, 0, 0),
      (0, 0, 0, 0, 0, 0, self.mixed, 0), (0, 0, 0, 0, 0, 0, 0, self.trivial),
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
          is_flip, tx, ty, k_rotate, is_gauss, is_log, is_mixed, is_trivial)
      transformation_list.append(transformation)

    self._transformation_ops = transformation_list


def test_visualize_transforms():
  import imageio
  import glob
  import os, sys
  import matplotlib.pyplot as plt

  im_path = os.path.join(PROJECT_PATH, 'extra_files', 'dragon.png')

  im = imageio.imread(im_path)

  im = im[np.newaxis, :150, :150, :3]
  im = im / np.max(im)
  print(im.shape)
  plt.imshow(im[0])
  plt.show()

  transformer = RankingTransformerOld()
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