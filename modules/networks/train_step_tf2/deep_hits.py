from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.utils import delta_timer
import time
import numpy as np


class DeepHits(tf.keras.Model):
  def __init__(self, input_shape, n_classes, drop_rate=0.0,
      final_activation='softmax', name='deep_hits', **kwargs):
    super().__init__(name=name)
    self.inp_shape = input_shape
    self.zp = tf.keras.layers.ZeroPadding2D(padding=(3, 3))
    self.conv_1 = tf.keras.layers.Conv2D(
        32, (4, 4), strides=(1, 1), padding='valid', activation='relu')
    self.conv_2 = tf.keras.layers.Conv2D(
        32, (3, 3), strides=(1, 1), padding='same', activation='relu')
    self.mp_1 = tf.keras.layers.MaxPool2D()
    self.conv_3 = tf.keras.layers.Conv2D(
        64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    self.conv_4 = tf.keras.layers.Conv2D(
        64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    self.conv_5 = tf.keras.layers.Conv2D(
        64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    self.mp_2 = tf.keras.layers.MaxPool2D()
    self.flatten = tf.keras.layers.Flatten()
    self.dense_1 = tf.keras.layers.Dense(64, activation='relu')
    self.do_1 = tf.keras.layers.Dropout(drop_rate)
    self.dense_2 = tf.keras.layers.Dense(64, activation='relu')
    self.do_2 = tf.keras.layers.Dropout(drop_rate)
    self.dense_3 = tf.keras.layers.Dense(n_classes)
    self.act_out = tf.keras.layers.Activation(final_activation)
    # builds
    self.loss_object = tf.keras.losses.CategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.Adam()
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='train_accuracy')
    self.val_loss = tf.keras.metrics.Mean(name='val_loss')
    self.val_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='val_accuracy')

  def call(self, input_tensor, training=False):
    x = self.zp(input_tensor)
    x = self.conv_1(x)
    x = self.conv_2(x)
    x = self.mp_1(x)
    x = self.conv_3(x)
    x = self.conv_4(x)
    x = self.conv_5(x)
    x = self.mp_2(x)
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.do_1(x, training=training)
    x = self.dense_2(x)
    x = self.do_2(x, training=training)
    x = self.dense_3(x)
    x = self.act_out(x)
    return x

  def model(self):
    x = tf.keras.layers.Input(shape=self.inp_shape)
    return tf.keras.Model(inputs=x, outputs=self.call(x))

  @tf.function
  def train_step(self, images, labels):
    with tf.GradientTape() as tape:
      predictions = self.call(images, training=True)
      loss = self.loss_object(labels, predictions)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    # self.train_loss(loss)
    # self.train_accuracy(labels, predictions)

  @tf.function
  def eval_step(self, images, labels, eval_loss, eval_acc):
    predictions = self.call(images, training=False)
    t_loss = self.loss_object(labels, predictions)
    eval_loss(t_loss)
    eval_acc(labels, predictions)

  def fit_tf(self, x, y, validation_data=None, batch_size=128, epochs=1,
      **kwargs):

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x, y)).shuffle(10000).batch(batch_size)
    for epoch in range(epochs):
      start_time = time.time()
      for images, labels in train_ds:
        self.train_step(images, labels)
      if kwargs['verbose']:
        self.eval_step(images, labels, self.train_loss,
                       self.train_accuracy)
        template = 'Epoch {}, Loss: {}, Acc: {}, Time: {}'
        print(template.format(epoch + 1,
                              self.train_loss.result(),
                              self.train_accuracy.result() * 100,
                              delta_timer(time.time() - start_time)
                              ))
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.eval_tf(x,y, verbose=kwargs['verbose'])

  def fit_tf_val(self, x, y, validation_data=None, batch_size=128, epochs=1,
      iterations_to_validate=None, **kwargs):
    if iterations_to_validate is None:
      iterations_to_validate = len(y)//batch_size
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x, y)).shuffle(10000).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (validation_data[0], validation_data[1])).batch(1024)
    for epoch in range(epochs):
      start_time = time.time()
      for it_i, (images, labels) in enumerate(train_ds):
        self.train_step(images, labels)
        if it_i%iterations_to_validate==0:
          for test_images, test_labels in val_ds:
            self.eval_step(test_images, test_labels, self.val_loss,
                           self.val_accuracy)
          self.eval_step(images, labels, self.train_loss,
                         self.train_accuracy)
          if kwargs['verbose']:
            template = 'Epoch {}, Loss: {}, Acc: {}, Val loss: {}, Val acc: {}, Time: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result() * 100,
                                  self.val_loss.result(),
                                  self.val_accuracy.result() * 100,
                                  delta_timer(time.time() - start_time)
                                  ))
          self.train_loss.reset_states()
          self.train_accuracy.reset_states()
          self.val_loss.reset_states()
          self.val_accuracy.reset_states()

  def predict_tf(self, x, batch_size=1024, **kwargs):
    eval_ds = tf.data.Dataset.from_tensor_slices((x)).batch(batch_size)
    predictions = []
    for images in eval_ds:
      predictions.append(self.call(images))
    return np.concatenate(predictions, axis=0)

  def eval_tf(self, x, y, batch_size=1024, **kwargs):
    eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    eval_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='eval_accuracy')
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x, y)).shuffle(10000).batch(batch_size)
    start_time = time.time()
    for images, labels in train_ds:
      self.eval_step(images, labels, eval_loss, eval_accuracy)
    if kwargs['verbose']:
      template = 'Loss: {}, Acc: {}, Time: {}'
      print(template.format(
          eval_loss.result(),
          eval_accuracy.result() * 100,
          delta_timer(time.time() - start_time)
      ))


if __name__ == '__main__':
  from parameters import loader_keys, general_keys
  from modules.geometric_transform import transformations_tf
  from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
  from tensorflow.keras.utils import to_categorical

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  hits_params = {
    loader_keys.DATA_PATH: os.path.join(
        PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
    loader_keys.N_SAMPLES_BY_CLASS: 10000,
    loader_keys.TEST_PERCENTAGE: 0.2,
    loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
    loader_keys.USED_CHANNELS: [2],
    loader_keys.CROP_SIZE: 21,
    general_keys.RANDOM_SEED: 42,
    loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
  }
  data_loader = HiTSOutlierLoader(hits_params)
  (x_train, y_train), (x_val, y_val), (
    x_test, y_test) = data_loader.get_outlier_detection_datasets()
  # transformer = transformations_tf.KernelTransformer(
  #     flips=True, gauss=False, log=False)
  transformer = transformations_tf.Transformer()
  x_train_transformed, transformations_inds = transformer.apply_all_transforms(
      x_train)
  x_val_transformed, transformations_inds_val = transformer.apply_all_transforms(
      x_val)
  mdl = DeepHits(input_shape=x_train.shape[1:],
                 n_classes=transformer.n_transforms)
  mdl.fit_tf(x_train_transformed, to_categorical(transformations_inds),
             verbose=1, iterations_to_validate=100,
             epochs=2, batch_size=128, validation_data=(
      x_val_transformed, to_categorical(transformations_inds_val)))
  mdl.eval_tf(x_train_transformed, to_categorical(transformations_inds),
              verbose=1)
  # mdl.save_weights('dummy.h5')
  # print(mdl.layers)
  # del mdl
  # mdl = DeepHits(input_shape=x_train.shape[1:],
  #                n_classes=transformer.n_transforms)
  # # mdl.build((None,) + x_train.shape[1:])
  # print(mdl.layers)
  # mdl.eval_tf(x_train_transformed, to_categorical(transformations_inds),
  #             verbose=1)
  # mdl.load_weights('dummy.h5')
  # mdl.eval_tf(x_train_transformed, to_categorical(transformations_inds),
  #             verbose=1)
