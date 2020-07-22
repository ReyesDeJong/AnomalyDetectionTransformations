import abc
import itertools

import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""There is a small discrepancy between original and his transformer, 
due the fact that padding reflects wiithouy copying the edge pixels"""


def cnn2d_depthwise_tf(image_batch, filters):
  df, _, cin, cmul = filters.shape
  padding = df // 2
  batch_padded = tf.pad(image_batch,
                        [[0, 0], [padding, padding],
                         [padding, padding],
                         [0, 0]], "REFLECT")
  features_tf = tf.nn.depthwise_conv2d(batch_padded, filters,
                                       strides=[1, 1, 1, 1],
                                       padding='VALID')

  return features_tf


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


def apply_affine_transform(res_x, t_x, t_y):
  # this are inverted, because to perform as train_step_tf2, we need to invert them
  tx = t_y
  ty = t_x
  res_x_padded = tf.pad(res_x,
                        [[0, 0], [tf.abs(ty), tf.abs(ty)],
                         [tf.abs(tx), tf.abs(tx)],
                         [0, 0]], "REFLECT")
  res_x_translated = res_x_padded
  res_x = res_x_translated[:,
          tf.abs(ty) + ty:tf.abs(ty) + ty +
                          res_x.shape[1],
          tf.abs(tx) + tx:tf.abs(tx) + tx +
                          res_x.shape[2], :]
  return res_x


# TODO: check if avoid doing this and include channel as
#  a transformator parameter speed ups things
def check_shape_kernel(kernel, x):
  if len(kernel.shape) == 2:
    kernel = tf.stack([kernel] * x.shape[-1], axis=-1)
    return tf.expand_dims(kernel, axis=-1)
  elif len(kernel.shape) == 3:
    return tf.expand_dims(kernel, axis=-1)
  return kernel


class KernelTransformation(object):
  def __init__(self, flip, tx, ty, k_90_rotate, gauss, log):
    self.flip = flip
    self.tx = tx
    self.ty = ty
    self.k_90_rotate = k_90_rotate
    self.gauss = gauss
    self.log = log
    self.gauss_kernel = makeGaussian(5, 1).astype(np.float32)
    self.log_kernel = makeLoG(5, 0.5).astype(np.float32)

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

    return res_x


# TODO: refactor translation, but test if its correctly done and speed up things or not, test tf.function
class AffineTransformation(object):
  def __init__(self, flip, tx, ty, k_90_rotate):
    """tx and ty are inverted to match original transformer"""
    self.flip = flip
    self.tx = ty
    self.ty = tx
    self.k_90_rotate = k_90_rotate

  def __call__(self, x):
    res_x = x
    if self.flip:
      with tf.name_scope("flip"):
        res_x = tf.image.flip_left_right(res_x)
    if self.tx != 0 or self.ty != 0:
      with tf.name_scope("translation"):
        res_x_padded = tf.pad(res_x,
                              [[0, 0], [np.abs(self.ty), np.abs(self.ty)],
                               [np.abs(self.tx), np.abs(self.tx)],
                               [0, 0]], "REFLECT")
        res_x_translated = res_x_padded
        res_x = res_x_translated[:,
                np.abs(self.ty) + self.ty:np.abs(self.ty) + self.ty +
                                          res_x.shape[1],
                np.abs(self.tx) + self.tx:np.abs(self.tx) + self.tx +
                                          res_x.shape[2], :]
    if self.k_90_rotate != 0:
      with tf.name_scope("rotation"):
        res_x = tf.image.rot90(res_x, k=self.k_90_rotate)
    return res_x


class AbstractTransformer(abc.ABC):
  def __init__(self, transform_batch_size=512, name='Abstract_Transformer'):
    self.name = name
    self._transform_batch_size = transform_batch_size
    # TODO: get none from returns
    self._transformation_ops = None
    self.transformation_tuples = None
    self._create_transformation_tuples_list()
    self._create_transformation_op_list()
    self.verbose = 1
    self.return_data_not_transformed = False

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

  def set_verbose(self, verbose_value):
    self.verbose = verbose_value

  def set_return_data_not_transformed(self, return_data_not_transformed):
    self.return_data_not_transformed = return_data_not_transformed

  def get_not_transformed_data_len(self, data_len):
    # TODO: NOT optimal because it requires correct transform batch size
    if self.return_data_not_transformed:
      return int(data_len / self.n_transforms)
    return data_len

  # This must be included within preprocessing mapping(?)
  # TODO: refactor transform batch to avoid appending
  def transform_batch(self, x, t_inds):
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
      transformed_batch = self.transform_batch(images, transformations_inds)
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


# ToDO: be more consistent on the usage of transform_batch_size
class Transformer(AbstractTransformer):
  def __init__(self, translation_x=8, translation_y=8,
      transform_batch_size=512, name='Transformer'):
    self.max_tx = translation_x
    self.max_ty = translation_y
    super().__init__(transform_batch_size, name)

  def _create_transformation_tuples_list(self):
    self.transformation_tuples = list(itertools.product((False, True),
                                                        (0, -self.max_tx,
                                                         self.max_tx),
                                                        (0, -self.max_ty,
                                                         self.max_ty),
                                                        range(4)))

  def _create_transformation_op_list(self):
    transformation_list = []
    for is_flip, tx, ty, k_rotate in self.transformation_tuples:
      transformation = AffineTransformation(is_flip, tx, ty, k_rotate)
      transformation_list.append(transformation)

    self._transformation_ops = transformation_list


class SimpleTransformer(AbstractTransformer):
  def __init__(self, transform_batch_size=512, name='Simple_Transformer'):
    super().__init__(transform_batch_size, name)

  def _create_transformation_tuples_list(self):
    self.transformation_tuples = list(itertools.product((False, True),
                                                        range(4)))

  def _create_transformation_op_list(self):
    transformation_list = []
    for is_flip, k_rotate in self.transformation_tuples:
      transformation = AffineTransformation(is_flip, 0, 0, k_rotate)
      transformation_list.append(transformation)

    self._transformation_ops = transformation_list


class TransTransformer(AbstractTransformer):
  def __init__(self, translation_x=8, translation_y=8,
      transform_batch_size=512, name='Trans_Transformer'):
    self.max_tx = translation_x
    self.max_ty = translation_y
    super().__init__(transform_batch_size, name)

  def _create_transformation_tuples_list(self):
    self.transformation_tuples = list(itertools.product(
        (0, -self.max_tx, self.max_tx),
        (0, -self.max_ty, self.max_ty),
    ))

  def _create_transformation_op_list(self):
    transformation_list = []
    for tx, ty in self.transformation_tuples:
      transformation = AffineTransformation(False, tx, ty, 0)
      transformation_list.append(transformation)

    self._transformation_ops = transformation_list


class KernelTransformer(AbstractTransformer):
  def __init__(self, translation_x=8, translation_y=8, rotations=False,
      flips=False, gauss=True, log=True, transform_batch_size=512,
      name='Kernel_Transformer'):
    self.iterable_tx = self.get_translation_iterable(translation_x)
    self.iterable_ty = self.get_translation_iterable(translation_y)
    self.iterable_rot = self.get_rotation_iterable(rotations)
    self.iterable_flips = self.get_bool_iterable(flips)
    self.iterable_gauss = self.get_bool_iterable(gauss)
    self.iterable_log = self.get_bool_iterable(log)
    super().__init__(transform_batch_size, name)

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

  def _create_transformation_tuples_list(self):
    self.transformation_tuples = list(itertools.product(
        self.iterable_flips,
        self.iterable_tx,
        self.iterable_ty,
        self.iterable_rot,
        self.iterable_gauss,
        self.iterable_log))

  def _create_transformation_op_list(self):
    transformation_list = []
    for is_flip, tx, ty, k_rotate, is_gauss, is_log in self.transformation_tuples:
      transformation = KernelTransformation(is_flip, tx, ty, k_rotate, is_gauss,
                                            is_log)
      transformation_list.append(transformation)

    self._transformation_ops = transformation_list


# TODO: see if can do some refactoring here
class PlusKernelTransformer(KernelTransformer):
  def __init__(self, translation_x=8, translation_y=8, rotations=True,
      flips=True, gauss=True, log=True, transform_batch_size=512,
      name='PlusKernel_Transformer'):
    super().__init__(translation_x, translation_y, rotations,
                     flips, gauss, log, transform_batch_size, name)

  def _create_transformation_tuples_list(self):
    self.transformation_tuples = list(itertools.product(
        self.iterable_flips,
        self.iterable_tx,
        self.iterable_ty,
        self.iterable_rot,
        [0],
        [0]))
    self.transformation_tuples += list(itertools.product(
        [0],
        self.iterable_tx,
        self.iterable_ty,
        [0],
        [1],
        [0]))
    self.transformation_tuples += list(itertools.product(
        [0],
        self.iterable_tx,
        self.iterable_ty,
        [0],
        [0],
        [1]))
    self.transformation_tuples += list(itertools.product(
        [0],
        self.iterable_tx,
        self.iterable_ty,
        [0],
        [1],
        [1]))

  def _create_transformation_op_list(self):
    transformation_list = []
    for is_flip, tx, ty, k_rotate, is_gauss, is_log in self.transformation_tuples:
      transformation = KernelTransformation(is_flip, tx, ty, k_rotate, is_gauss,
                                            is_log)
      transformation_list.append(transformation)

    self._transformation_ops = transformation_list


# TODO: see if can do some refactoring here
class PlusGaussTransformer(KernelTransformer):
  def __init__(self, translation_x=8, translation_y=8, rotations=True,
      flips=True, gauss=True, log=True, transform_batch_size=512,
      name='PlusGauss_Transformer'):
    super().__init__(translation_x, translation_y, rotations,
                     flips, gauss, log, transform_batch_size, name)

  def _create_transformation_tuples_list(self):
    self.transformation_tuples = list(itertools.product(
        self.iterable_flips,
        self.iterable_tx,
        self.iterable_ty,
        self.iterable_rot,
        [0],
        [0]))
    self.transformation_tuples += list(itertools.product(
        [0],
        self.iterable_tx,
        self.iterable_ty,
        [0],
        [1],
        [0]))

  def _create_transformation_op_list(self):
    transformation_list = []
    for is_flip, tx, ty, k_rotate, is_gauss, is_log in self.transformation_tuples:
      transformation = KernelTransformation(is_flip, tx, ty, k_rotate, is_gauss,
                                            is_log)
      transformation_list.append(transformation)

    self._transformation_ops = transformation_list


# TODO: see if can do some refactoring here
class PlusLaplaceTransformer(KernelTransformer):
  def __init__(self, translation_x=8, translation_y=8, rotations=True,
      flips=True, gauss=True, log=True, transform_batch_size=512,
      name='PlusLaplace_Transformer'):
    super().__init__(translation_x, translation_y, rotations,
                     flips, gauss, log, transform_batch_size, name)

  def _create_transformation_tuples_list(self):
    self.transformation_tuples = list(itertools.product(
        self.iterable_flips,
        self.iterable_tx,
        self.iterable_ty,
        self.iterable_rot,
        [0],
        [0]))
    self.transformation_tuples += list(itertools.product(
        [0],
        self.iterable_tx,
        self.iterable_ty,
        [0],
        [0],
        [1]))

  def _create_transformation_op_list(self):
    transformation_list = []
    for is_flip, tx, ty, k_rotate, is_gauss, is_log in self.transformation_tuples:
      transformation = KernelTransformation(is_flip, tx, ty, k_rotate, is_gauss,
                                            is_log)
      transformation_list.append(transformation)

    self._transformation_ops = transformation_list


def test_visualize_transforms():
  import imageio
  import glob
  import os, sys
  import matplotlib.pyplot as plt

  PROJECT_PATH = os.path.abspath(
      os.path.join(os.path.dirname(__file__), '..', '..'))
  sys.path.append(PROJECT_PATH)

  im_path = os.path.join(PROJECT_PATH, 'extra_files', 'dragon.png')

  for im_path in glob.glob(im_path):
    im = imageio.imread(im_path)

  im = im[np.newaxis, :150, :150, :]
  im = im / np.max(im)
  print(im.shape)
  plt.imshow(im[0])
  plt.show()

  transformer = Transformer(60, 60)
  transformations_inds = np.arange(transformer.n_transforms)

  transformed_batch = transformer.transform_batch(im,
                                                  transformations_inds)

  print(transformed_batch.shape)

  for i in range(72):
    transform_indx = i
    if (i % 4) == 0 or i == 1 or i == 2 or i == 3:
      plt.imshow(transformed_batch[transform_indx])
      plt.title(str(transformer.transformation_tuples[i]))
      plt.axis('off')
      plt.show()


def plot_img(transformed_batch, transformer, indx, batch_size=8):
  import matplotlib.pyplot as plt
  plt.imshow(transformed_batch.numpy()[indx])
  transform_indx = indx // batch_size
  plt.title(str(transformer.transformation_tuples[transform_indx]))
  plt.show()


def test_dataset_generation():
  import imageio
  import glob
  import os, sys
  import datetime
  import time

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  PROJECT_PATH = os.path.abspath(
      os.path.join(os.path.dirname(__file__), '..', '..'))
  sys.path.append(PROJECT_PATH)

  im_path = os.path.join(PROJECT_PATH, 'extra_files', 'dragon.png')

  for im_path in glob.glob(im_path):
    im = imageio.imread(im_path)

  im = im[np.newaxis, 5:68, 5:68, :]
  im = im / np.max(im)
  print(im.shape)
  # plt.imshow(im[0])
  # plt.show()

  dataset_size = 1000
  batch_size = 32
  x_test = np.repeat(im, dataset_size, axis=0)
  test_ds = tf.data.Dataset.from_tensor_slices((x_test)).batch(batch_size)
  transformer = Transformer(8, 8)
  transformations_inds = np.arange(transformer.n_transforms)

  EPOCHS = 1

  start_time = time.time()
  for epoch in range(EPOCHS):
    transformed_dataset = []
    for images in test_ds:
      transformed_batch = transformer.transform_batch(images,
                                                      transformations_inds)
      transformed_dataset.append(transformed_batch)
      # print(transformed_batch.shape)
    transformed_dataset = np.concatenate(
        [tensor.numpy() for tensor in transformed_dataset])
  print(transformed_dataset.shape)

  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time usage %s: %s" % (transformer.name, str(time_usage)), flush=True)

  last_batch_size = dataset_size % batch_size
  indx_to_plot = np.arange(transformer.n_transforms) * last_batch_size
  for i in indx_to_plot:
    plot_img(transformed_batch, transformer, i)


def plot_astro_img(x, transform):
  import matplotlib.pyplot as plt
  plt.imshow(x[..., 0])
  plt.title(str(transform))
  plt.show()


if __name__ == "__main__":
  test_visualize_transforms()
  # import imageio
  # import glob
  # import os, sys
  # import matplotlib.pyplot as plt
  # import datetime
  # import time
  #
  # PROJECT_PATH = os.path.abspath(
  #     os.path.join(os.path.dirname(__file__), '..', '..'))
  # sys.path.append(PROJECT_PATH)
  #
  # from modules.utils import set_soft_gpu_memory_growth
  # from parameters import loader_keys, general_keys
  # from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
  #
  # set_soft_gpu_memory_growth()
  #
  # params = {
  #   loader_keys.DATA_PATH: os.path.join(
  #       PROJECT_PATH, '../datasets/ztf_v1_bogus_added.pkl'),
  #   loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
  #   loader_keys.USED_CHANNELS: [0, 1, 2],
  #   loader_keys.CROP_SIZE: 21,
  #   general_keys.RANDOM_SEED: 42,
  #   loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  # }
  # transformer = Transformer()
  # batch_size = 512
  # ztf_outlier_dataset = ZTFOutlierLoader(params)
  # (X_train, y_train), (X_val, y_val), (
  #   X_test, y_test) = ztf_outlier_dataset.get_outlier_detection_datasets()
  #
  # train_ds = tf.data.Dataset.from_tensor_slices((X_train)).batch(batch_size)
  # transformations_inds = np.arange(transformer.n_transforms)
  #
  # EPOCHS = 1
  #
  # # # no labels
  # # train_ds = tf.data.Dataset.from_tensor_slices((X_train)).batch(batch_size)
  # # transformations_inds = np.arange(transformer.n_transforms)
  # # start_time = time.time()
  # # for epoch in range(EPOCHS):
  # #   transformed_dataset_v0 = []
  # #   for images in train_ds:
  # #     transformed_batch = transformer.transform_batch(images,
  # #                                                     transformations_inds)
  # #     transformed_dataset_v0.append(transformed_batch)
  # #   transformed_dataset_v0 = np.concatenate(
  # #       [tensor.numpy() for tensor in transformed_dataset_v0])
  # # print(transformed_dataset_v0.shape)
  # # time_usage = str(datetime.timedelta(
  # #     seconds=int(round(time.time() - start_time))))
  # # print("Time usage No Labels %s: %s" % (transformer.name, str(time_usage)),
  # #       flush=True)
  # #
  # # retrieving labels along side batch generationg
  # start_time = time.time()
  # for epoch in range(EPOCHS):
  #   transformed_dataset_v0 = transformer.apply_all_transformsv0(X_train,
  #                                                               batch_size)
  # print(transformed_dataset_v0[0].shape)
  # time_usage = str(datetime.timedelta(
  #     seconds=int(round(time.time() - start_time))))
  # print("Time usage Images along labels v0 %s: %s" % (
  #   transformer.name, str(time_usage)), flush=True)
  #
  # # retrieving labels after batch generationg
  # start_time = time.time()
  # for epoch in range(EPOCHS):
  #   transformed_dataset_v1 = transformer.apply_all_transforms(X_train,
  #                                                             batch_size)
  # print(transformed_dataset_v1[0].shape)
  # time_usage = str(datetime.timedelta(
  #     seconds=int(round(time.time() - start_time))))
  # print("Time usage Images along labels v1 %s: %s" % (
  #   transformer.name, str(time_usage)), flush=True)
  #
  # start_time = time.time()
  # (X_train, y_train), (X_val, y_val), (
  #   X_test, y_test) = ztf_outlier_dataset.get_transformed_datasets(
  #     transformer)
  # print("Time usage Loading Pickle %s: %s" % (
  #   transformer.name, str(time_usage)), flush=True)
  #
  # # from transformations import Transformer as slow_Transformer
  # #
  # # slow_transformer = slow_Transformer()
  # # (X_train, y_train), (X_val, y_val), (
  # #   X_test, y_test) = ztf_outlier_dataset.get_transformed_datasets(
  # #     slow_transformer)
  # #
  # # # tranform + n_transforms * sample_i_in_one_batch
  # # sample_i_in_one_batch = 158
  # # transform_i = 60
  # # plot_astro_img(X_train[transform_i + 72 * sample_i_in_one_batch],
  # #                transformer.transformation_tuples[
  # #                  y_train[transform_i + 72 * sample_i_in_one_batch]])
  # # # sample_i_in_one_batch + tranform * batch_size
  # # plot_astro_img(transformed_dataset_v1[0][
  # #                  sample_i_in_one_batch + transform_i * batch_size],
  # #                transformer.transformation_tuples[
  # #                  transformed_dataset_v1[1][
  # #                    sample_i_in_one_batch + transform_i * batch_size]])
