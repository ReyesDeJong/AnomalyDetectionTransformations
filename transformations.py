import abc
import itertools

import numpy as np
import tensorflow as tf
import torch
from keras import backend as K
from keras.preprocessing.image import apply_affine_transform
from torch.nn import functional as F


def createCircularMask(h, w, center=None, radius=None):
  if center is None:  # use the middle of the image
    center = [int(w / 2), int(h / 2)]
  if radius is None:  # use the smallest distance between the center and image walls
    radius = min(center[0], center[1], w - center[0], h - center[1])

  Y, X = np.ogrid[:h, :w]
  dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

  mask = dist_from_center <= radius
  return mask*1.0


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


def depth_wise_conv2d_tf(x, kernel):
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


def convert_to_torch(image, filters):
  image_torch = torch.tensor(image.transpose([2, 1, 0])[None])
  filters_torch = torch.tensor(filters.transpose([3, 2, 1, 0]))

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

  features_torch = F.conv2d(image_torch, filters_torch, padding=df // 2,
                            groups=cin)
  features_torch_ = features_torch.cpu().numpy()[0].transpose([2, 1, 0])

  return features_torch_


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from keras.backend.tensorflow_backend import set_session
  import datetime
  import time

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


  # g1 = makeGaussian(21, 1)
  # g2 = makeGaussian(21, 2)
  # g3 = makeGaussian(21, 1, (3, 3))
  g1 = createCircularMask(21, 21, radius=4)
  g2 = createCircularMask(21, 21, radius=5)
  g3 = createCircularMask(21, 21, [3, 3], radius=3)
  gauss_image = np.stack([g1, g2, g3], axis=-1)
  plot_image(gauss_image)
  gauss_kernel = np.stack([makeGaussian(5, 1)] * 3, axis=-1)
  plot_image(gauss_kernel)

  tf_convolved = depth_wise_conv2d_tf(gauss_image, gauss_kernel)
  plot_image(tf_convolved)

  torch_convolved = cnn2d_depthwise_torch(gauss_image,
                                          check_shape_kernel(gauss_kernel,
                                                             gauss_image))
  plot_image(torch_convolved)

  print('difference between pytorch and tf ',
        np.mean(tf_convolved - torch_convolved))



  iters=500

  start_time = time.time()
  for i in range(iters):
    torch_convolved = cnn2d_depthwise_torch(gauss_image,
                                            check_shape_kernel(gauss_kernel,
                                                               gauss_image))
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time usage Torch: " + time_usage, flush=True)

  start_time = time.time()
  for i in range(iters):
    tf_convolved = depth_wise_conv2d_tf(gauss_image, gauss_kernel)
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time usage Keras: " + time_usage, flush=True)





