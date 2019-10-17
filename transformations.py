import abc
import itertools

import numpy as np
import scipy.stats as st
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import apply_affine_transform


def gkern(kernlen=5, nsig=2):
  """Returns a 2D Gaussian kernel."""

  x = np.linspace(-nsig, nsig, kernlen + 1)
  kern1d = np.diff(st.norm.cdf(x))
  kern2d = np.outer(kern1d, kern1d)
  return kern2d / kern2d.sum()


def np_gkern(kernlen=5, nsig=2):
  x, y = np.meshgrid(np.linspace(-1, 1, kernlen), np.linspace(-1, 1, kernlen))
  d = np.sqrt(x * x + y * y)
  sigma, mu = nsig, 0.0
  g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
  return g


def check_shape_image(image):
  if len(image.shape) == 2:
    return image[np.newaxis, ..., np.newaxis]
  elif len(image.shape) == 3 and image.shape[-1] != image.shape[-2]:
    return image[np.newaxis, ...]
  elif len(image.shape) == 3 and image.shape[0] != image.shape[1]:
    return image[..., np.newaxis]
  return image


def check_shape_kernel(kernel, x):
  if len(kernel.shape) == 2:
    kernel = np.stack([kernel]*x.shape[-1], axis=-1)
    return kernel[..., np.newaxis]
  return kernel


def depth_wise_conv2d(x, kernel):
  """
  perform convolution on each channel of x by separate.
  x is a single image of shape HWC
  :param x:
  :param kernel:
  :return:
  """
  # x_channel_split = np.split(x, x.shape[-1], -1)
  # x_conv2d_list = [K.eval(
  #   tf.nn.conv2d(check_shape_image(x_i), check_shape_kernel(kernel),
  #                padding='SAME'))[..., 0] for x_i in
  #                  x_channel_split]
  # return np.stack(x_conv2d_list, axis=-1)[0]
  return K.eval(
      tf.nn.depthwise_conv2d(check_shape_image(x),
                             check_shape_kernel(kernel, x),
                             (1, 1, 1, 1),
                             padding='SAME'))[0]


class AffineTransformation(object):
  def __init__(self, flip, tx, ty, k_90_rotate):
    self.flip = flip
    self.tx = tx
    self.ty = ty
    self.k_90_rotate = k_90_rotate

  def __call__(self, x):
    res_x = x
    if self.flip:
      res_x = np.fliplr(res_x)
    if self.tx != 0 or self.ty != 0:
      res_x = apply_affine_transform(res_x, tx=self.tx, ty=self.ty,
                                     channel_axis=2, fill_mode='reflect')
    if self.k_90_rotate != 0:
      res_x = np.rot90(res_x, self.k_90_rotate)

    return res_x


class AbstractTransformer(abc.ABC):
  def __init__(self):
    self._transformation_list = None
    self._create_transformation_list()

  @property
  def n_transforms(self):
    return len(self._transformation_list)

  @abc.abstractmethod
  def _create_transformation_list(self):
    return

  def transform_batch(self, x_batch, t_inds):
    assert len(x_batch) == len(t_inds)

    transformed_batch = x_batch.copy()
    for i, t_ind in enumerate(t_inds):
      transformed_batch[i] = self._transformation_list[t_ind](
          transformed_batch[i])
    return transformed_batch


class Transformer(AbstractTransformer):
  def __init__(self, translation_x=8, translation_y=8):
    self.max_tx = translation_x
    self.max_ty = translation_y
    super().__init__()

  def _create_transformation_list(self):
    transformation_list = []
    for is_flip, tx, ty, k_rotate in itertools.product((False, True),
                                                       (0, -self.max_tx,
                                                        self.max_tx),
                                                       (0, -self.max_ty,
                                                        self.max_ty),
                                                       range(4)):
      transformation = AffineTransformation(is_flip, tx, ty, k_rotate)
      transformation_list.append(transformation)

    self._transformation_list = transformation_list


class SimpleTransformer(AbstractTransformer):
  def _create_transformation_list(self):
    transformation_list = []
    for is_flip, k_rotate in itertools.product((False, True),
                                               range(4)):
      transformation = AffineTransformation(is_flip, 0, 0, k_rotate)
      transformation_list.append(transformation)

    self._transformation_list = transformation_list


class TransTransformer(AbstractTransformer):
  def __init__(self, translation_x=8, translation_y=8):
    self.max_tx = translation_x
    self.max_ty = translation_y
    super().__init__()

  def _create_transformation_list(self):
    transformation_list = []
    for tx, ty in itertools.product(
        (0, -self.max_tx, self.max_tx),
        (0, -self.max_ty, self.max_ty),
    ):
      transformation = AffineTransformation(False, tx, ty, 0)
      transformation_list.append(transformation)

    self._transformation_list = transformation_list
