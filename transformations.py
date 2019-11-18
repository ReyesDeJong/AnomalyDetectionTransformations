import abc
import itertools

import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.preprocessing.image import apply_affine_transform
from torch.nn import functional as F
from tqdm import tqdm


def convert_to_torch(image, filters):
  image_torch = torch.tensor(image.transpose([2, 1, 0])[None]).float()
  filters_torch = torch.tensor(filters.transpose([3, 2, 1, 0])).float()

  return image_torch, filters_torch


def cnn2d_depthwise_torch(image: np.ndarray,
    filters: np.ndarray):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  image_torch, filters_torch = convert_to_torch(image, filters)
  image_torch, filters_torch = image_torch.to(device), filters_torch.to(device)

  df, _, cin, cmul = filters.shape
  filters_torch = filters_torch.transpose(0, 1).contiguous()
  filters_torch = filters_torch.view(cin * cmul, 1, df, df)
  # print(filters_torch.shape)
  features_torch = F.conv2d(image_torch, filters_torch, padding=df // 2,
                            groups=cin)
  features_torch_ = features_torch.cpu().numpy()[0].transpose([2, 1, 0])

  return features_torch_


def makeGaussian(size, sigma=3, center=None):
  """ Make a square gaussian kernel.

  size is the length of a side of the square
  fwhm is full-width-half-maximum, which
  can be thought of as an effective radius.
  """

  x = np.arange(0, size, 1, float)
  y = x[:, np.newaxis]

  if center is None:
    x0 = y0 = size // 2
  else:
    x0 = center[0]
    y0 = center[1]

  return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))


def makeLoG(size, sigma=3, center=None):
  """ Make a square LoG kernel.

  size is the length of a side of the square
  fwhm is full-width-half-maximum, which
  can be thought of as an effective radius.
  """

  x = np.arange(0, size, 1, float)
  y = x[:, np.newaxis]

  if center is None:
    x0 = y0 = size // 2
  else:
    x0 = center[0]
    y0 = center[1]

  return (-1 / (np.pi * sigma ** 4)) * (
      1 - (((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))) * np.exp(
      -((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))


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
    kernel = np.stack([kernel] * x.shape[-1], axis=-1)
    return kernel[..., np.newaxis]
  elif len(kernel.shape) == 3:
    return kernel[..., np.newaxis]
  return kernel


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


class KernelTransformation(object):
  def __init__(self, flip, tx, ty, k_90_rotate, gauss, log):
    self.flip = flip
    self.tx = tx
    self.ty = ty
    self.k_90_rotate = k_90_rotate
    self.gauss = gauss
    self.log = log
    self.gauss_kernel = makeGaussian(5, 1)
    self.log_kernel = makeLoG(5, 0.5)

  def __call__(self, x):
    res_x = x
    if self.gauss:
      res_x = cnn2d_depthwise_torch(
          res_x, check_shape_kernel(self.gauss_kernel, res_x))
    if self.log:
      res_x = cnn2d_depthwise_torch(
          res_x, check_shape_kernel(self.log_kernel, res_x))
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
    self.name = 'Abstract_transformer'

  @property
  def n_transforms(self):
    return len(self._transformation_list)

  @abc.abstractmethod
  def _create_transformation_list(self):
    return

  def transform_batch(self, x_batch, t_inds):
    assert len(x_batch) == len(t_inds)

    transformed_batch = x_batch.copy()
    for i in tqdm(range(len(t_inds))):
      transformed_batch[i] = self._transformation_list[t_inds[i]](
          transformed_batch[i])
    return transformed_batch


class Transformer(AbstractTransformer):
  def __init__(self, translation_x=8, translation_y=8):
    self.max_tx = translation_x
    self.max_ty = translation_y
    super().__init__()
    self.name = '72_transformer'

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
  def __init__(self):
    super().__init__()
    self.name = 'Rotate_transformer'

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
    self.name = 'Trans_transformer'

  def _create_transformation_list(self):
    transformation_list = []
    for tx, ty in itertools.product(
        (0, -self.max_tx, self.max_tx),
        (0, -self.max_ty, self.max_ty),
    ):
      transformation = AffineTransformation(False, tx, ty, 0)
      transformation_list.append(transformation)

    self._transformation_list = transformation_list


class KernelTransformer(AbstractTransformer):
  def __init__(self, translation_x=8, translation_y=8, rotations=True,
      flips=True, gauss=True, log=True):
    self.iterable_tx = self.get_translation_iterable(translation_x)
    self.iterable_ty = self.get_translation_iterable(translation_y)
    self.iterable_rot = self.get_rotation_iterable(rotations)
    self.iterable_flips = self.get_bool_iterable(flips)
    self.iterable_gauss = self.get_bool_iterable(gauss)
    self.iterable_log = self.get_bool_iterable(log)
    super().__init__()
    self.name = 'Kernel_transformer'

  def get_translation_iterable(self, translation):
    if translation:
      return (0, -translation, translation)
    return range(1)

  def get_rotation_iterable(self, rotations):
    if rotations:
      return range(4)
    return range(1)

  def get_bool_iterable(self, bool_variable):
    if bool_variable:
      return range(2)
    return range(1)

  def _create_transformation_list(self):
    transformation_list = []
    for is_flip, tx, ty, k_rotate, is_gauss, is_log in itertools.product(
        self.iterable_flips,
        self.iterable_tx,
        self.iterable_ty,
        self.iterable_rot,
        self.iterable_gauss,
        self.iterable_log):
      transformation = KernelTransformation(is_flip, tx, ty, k_rotate, is_gauss,
                                            is_log)
      transformation_list.append(transformation)

    self._transformation_list = transformation_list


if __name__ == "__main__":
  #tf 1
  import matplotlib.pyplot as plt
  from keras.backend.tensorflow_backend import set_session
  from modules.utils import createCircularMask

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
  sess = tf.Session(config=config)
  set_session(sess)


  def plot_image(image):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(image[..., 0])
    ax[1].imshow(image[..., 1])
    ax[2].imshow(image[..., 2])
    plt.show()


  g1 = createCircularMask(21, 21, radius=4)
  g2 = createCircularMask(21, 21, radius=5)
  g3 = createCircularMask(21, 21, [3, 3], radius=1)
  gauss_image = np.stack([g1, g2, g3], axis=-1)
  plot_image(gauss_image)
  gauss_kernel = np.stack([makeGaussian(5, 1)] * 3, axis=-1)
  log_kernel = np.stack([makeLoG(5, 0.5)] * 3, axis=-1)

  plot_image(gauss_kernel)
  plot_image(log_kernel)

  torch_convolved = cnn2d_depthwise_torch(gauss_image,
                                          check_shape_kernel(gauss_kernel,
                                                             gauss_image))
  plot_image(torch_convolved)

  torch_convolved = cnn2d_depthwise_torch(gauss_image,
                                          check_shape_kernel(log_kernel,
                                                             gauss_image))
  plot_image(torch_convolved)
