import abc
import itertools

import tensorflow as tf


class AffineTransformation(object):
  def __init__(self, flip, tx, ty, k_90_rotate):
    self.flip = flip
    self.tx = tx
    self.ty = ty
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
        res_x_translated = res_x_padded  # tf.contrib.image.translate(
        # res_x_padded, [self.tx, self.ty])
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

  # This must be included within preprocessing mapping(?)
  def transform_batch(self, x, t_inds):
    transformed_batch = []
    with tf.name_scope("transformations"):
      for i, t_ind in enumerate(t_inds):
        transformed_batch.append(self._transformation_list[t_ind](x))
      concatenated_transformations = tf.concat(transformed_batch, axis=0)
    return tf.identity(concatenated_transformations, 'concat_transforms')

  def apply_all_transforms(self, x):
    """generate transform inds, that are the labels of each transform and
    its respective transformed data"""
    # img batch of dim: [B, H, W, C]
    transformations_inds = np.arange(self.n_transforms)
    transformed_x = self.transform_batch(x, transformations_inds)
    return transformed_x, transformations_inds


class Transformer(AbstractTransformer):
  def __init__(self, translation_x=8, translation_y=8):
    self.max_tx = translation_x
    self.max_ty = translation_y
    super().__init__()
    self.name = '72_transformer'

  def _create_transformation_list(self):
    transformation_list = []
    self.tranformation_to_perform = list(itertools.product((False, True),
                                                           (0, -self.max_tx,
                                                            self.max_tx),
                                                           (0, -self.max_ty,
                                                            self.max_ty),
                                                           range(4)))
    for is_flip, tx, ty, k_rotate in self.tranformation_to_perform:
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

  transformer = Transformer(16, 16)
  transformations_inds = np.arange(transformer.n_transforms)

  transformed_batch = transformer.transform_batch(im,
                                                  transformations_inds)

  print(transformed_batch.shape)

  for i in range(72):
    transform_indx = i
    if (i % 4) == 0:
      plt.imshow(transformed_batch[transform_indx])
      plt.title(str(transformer.tranformation_to_perform[i]))
      plt.show()


if __name__ == "__main__":
  import imageio
  import glob
  import os, sys
  import matplotlib.pyplot as plt
  import datetime
  import time
  import numpy as np

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  PROJECT_PATH = os.path.abspath(
      os.path.join(os.path.dirname(__file__), '..', '..'))
  sys.path.append(PROJECT_PATH)

  im_path = os.path.join(PROJECT_PATH, 'extra_files', 'dragon.png')

  for im_path in glob.glob(im_path):
    im = imageio.imread(im_path)

  im = im[np.newaxis, :63, :63, :]
  im = im / np.max(im)
  print(im.shape)
  # plt.imshow(im[0])
  # plt.show()

  x_test = np.repeat(im, 1000, axis=0)
  test_ds = tf.data.Dataset.from_tensor_slices((x_test)).batch(32)
  transformer = Transformer(16, 16)
  transformations_inds = np.arange(transformer.n_transforms)

  EPOCHS = 2

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

  # for i in range(72):
  #   transform_indx = i
  #   if (i % 4) == 0:
  #     plt.imshow(transformed_batch[transform_indx])
  #     plt.title(str(transformer.tranformation_to_perform[i]))
  #     plt.show()
  #
