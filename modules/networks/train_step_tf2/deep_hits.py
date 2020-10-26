from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# PROJECT_PATH = os.path.join(PROJECT_PATH, os.path.abspath('../../../../../home/ereyes/Projects/Thesis/AnomalyDetectionTransformations'))
# print(os.path.abspath(PROJECT_PATH))
sys.path.append(PROJECT_PATH)


from modules.utils import delta_timer
import time
import numpy as np
from parameters import loader_keys, general_keys, constants
from modules import utils


class DeepHits(tf.keras.Model):
  def __init__(self, input_shape, n_classes, drop_rate=0.0,
      final_activation='softmax', name='deep_hits', model_path='', **kwargs):
    super().__init__(name=name)
    self.model_path = model_path
    # TODO: relgate this to method in order to make a unified init
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
    self._init_builds()
    self._keras_compile()

  def _init_builds(self):
    # builds
    self.loss_object = tf.keras.losses.CategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.Adam()
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='train_accuracy')
    self.eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    self.eval_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='eval_accuracy')

  def _keras_compile(self):
    self.compile(
          general_keys.ADAM, general_keys.CATEGORICAL_CROSSENTROPY,
          [general_keys.ACC])

  def call(self, input_tensor, training=False, remove_top=False):
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
    if remove_top:
      return x
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
    self.train_loss(loss)
    self.train_accuracy(labels, predictions)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

  @tf.function
  def eval_step(self, images, labels):
    predictions = self.call(images, training=False)
    t_loss = self.loss_object(labels, predictions)
    self.eval_loss(t_loss)
    self.eval_accuracy(labels, predictions)

  def fit(self, x, y, validation_data=None, batch_size=128, epochs=1,
      iterations_to_validate=None, verbose=1, patience=None, callbacks=None,
      **kwargs):
    if patience is None and callbacks is None:
      patience=0
    if callbacks and patience is None:
      patience=callbacks[0].patience
    print('\nTraining Model')
    self.training_star_time = time.time()
    self.verbose = verbose
    if validation_data is not None:
      return self.fit_tf_val(
          x, y, validation_data, batch_size, epochs,
          iterations_to_validate, patience, verbose)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x, y)).shuffle(10000).batch(batch_size)
    for epoch in range(epochs):
      epoch_start_time = time.time()
      for images, labels in train_ds:
        self.train_step(images, labels)
      if self.verbose:
        eval_results = self.eval_tf(x, y, verbose=False)
        template = 'Epoch {}, Loss: {}, Acc: {}, Time: {}'
        print(template.format(epoch + 1,
                              eval_results[general_keys.LOSS],
                              eval_results[general_keys.ACCURACY] * 100,
                              delta_timer(time.time() - epoch_start_time)
                              ))
    print('Total Training Time: {}'.format(
        delta_timer(time.time() - self.training_star_time)))

  def fit_tf_val(self, x, y, validation_data=None, batch_size=128, epochs=1,
      iterations_to_validate=None, patience=0, verbose=1):
    self.verbose = verbose
    n_iterations_in_epoch = (len(y) // batch_size)-1
    # check if validate at end of epoch
    if iterations_to_validate is None:
      iterations_to_validate = n_iterations_in_epoch
    self.best_model_so_far = {
      general_keys.ITERATION: 0,
      general_keys.LOSS: 1e100,
      general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH: 0,
    }
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x, y)).shuffle(10000).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (validation_data[0], validation_data[1])).batch(1024)
    self.check_best_model_save(iteration=0)
    start_time = time.time()
    for epoch in range(epochs):
      for it_i, (images, labels) in enumerate(train_ds):
        self.train_step(images, labels)
        # print(it_i)
        if it_i % iterations_to_validate == 0 and it_i != 0:
          # print('validate')
          if self.check_early_stopping(patience):
            return
          for test_images, test_labels in val_ds:
            self.eval_step(test_images, test_labels)
          # TODO: check train printer
          # print('hola')
          if self.verbose:
            # print('here')
            template = 'Epoch {}, Loss: {}, Acc: {}, Val loss: {}, Val acc: {}, Time: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result() * 100,
                                  self.eval_loss.result(),
                                  self.eval_accuracy.result() * 100,
                                  delta_timer(time.time() - start_time)
                                  ))
          self.check_best_model_save(
              it_i + ((epoch + 1) * n_iterations_in_epoch))
          self.eval_loss.reset_states()
          self.eval_accuracy.reset_states()
          self.train_loss.reset_states()
          self.train_accuracy.reset_states()
    self.load_weights(
        self.best_model_weights_path).expect_partial()
    print('Total Training Time: {}'.format(
        delta_timer(time.time() - self.training_star_time)))

  def check_early_stopping(self, patience):
    if self.best_model_so_far[
      general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] >= patience + 1:
      # print(self.best_model_weights_path)
      self.load_weights(
          self.best_model_weights_path).expect_partial()
      self.eval_loss.reset_states()
      self.eval_accuracy.reset_states()
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()
      print('Total Training Time: {}'.format(
          delta_timer(time.time() - self.training_star_time)))
      return True
    return False

  def check_best_model_save(self, iteration):
    if iteration == 0:
      best_model_weights_folder = os.path.join(self.model_path,
                                               'aux_weights')
      utils.check_path(best_model_weights_folder)
      self.best_model_weights_path = os.path.join(best_model_weights_folder,
                                                  'best_weights.ckpt')
      self.save_weights(self.best_model_weights_path)
      return
    self.best_model_so_far[general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] += 1
    if self.eval_loss.result() < self.best_model_so_far[general_keys.LOSS]:
      self.best_model_so_far[general_keys.LOSS] = self.eval_loss.result()
      self.best_model_so_far[general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] = 0
      self.best_model_so_far[general_keys.ITERATION] = iteration
      self.save_weights(self.best_model_weights_path)
      if self.verbose:
        print("\nNew best validation model: %s %.4f @ it %d\n" % (
          general_keys.LOSS,
          self.best_model_so_far[general_keys.LOSS],
          self.best_model_so_far[general_keys.ITERATION]), flush=True)

  @tf.function
  def call_wrapper_to_predict(self, x):
    return self.call(x)

  def predict(self, x, batch_size=1024):
    # if len(x)%batch_size!=0:
    #   #TODO: check memory leakage of appen, if it creates copy of array or not
    #   extras_x = np.zeros(((batch_size-(len(x)%batch_size),)+ x.shape[1:]))
    #   extras_x = np.append(x, extras_x, 0)
    #   extras_pred = super().predict(extras_x, batch_size)
    #   return extras_pred[:len(x)]
    # return super().predict(x, batch_size)
    eval_ds = tf.data.Dataset.from_tensor_slices((x)).batch(batch_size)
    predictions = []
    for images in eval_ds:
      predictions.append(self.call_wrapper_to_predict(images))
    return np.concatenate(predictions, axis=0)


  def eval_tf(self, x, y, batch_size=1024, verbose=1):
    self.eval_loss.reset_states()
    self.eval_accuracy.reset_states()
    dataset = tf.data.Dataset.from_tensor_slices(
        (x, y)).batch(batch_size)
    start_time = time.time()
    for images, labels in dataset:
      self.eval_step(images, labels)
    if verbose:
      template = 'Loss: {}, Acc: {}, Time: {}'
      print(template.format(
          self.eval_loss.result(),
          self.eval_accuracy.result() * 100,
          delta_timer(time.time() - start_time)
      ))
    results_dict = {general_keys.LOSS: self.eval_loss.result(),
                    general_keys.ACCURACY: self.eval_accuracy.result()}
    self.eval_loss.reset_states()
    self.eval_accuracy.reset_states()
    return results_dict

  @tf.function
  def get_activation_step(self, batch):
    return self.call(batch, remove_top=True)

  def get_activations(self, x, batch_size=1024):
    dataset = tf.data.Dataset.from_tensor_slices(
        (x)).batch(batch_size)
    activations_list = []
    for images in dataset:
      # print(images.shape)
      activations = self.get_activation_step(images)
      # print(predictions.shape)
      activations_list.append(activations)
    concatenated_activation = np.concatenate(activations_list)
    # print(concatenated_activation.shape)
    return concatenated_activation.reshape(concatenated_activation.shape[0], np.prod(concatenated_activation.shape[1:]))


if __name__ == '__main__':
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
  transformer = transformations_tf.TransTransformer()
  x_train_transformed, transformations_inds = transformer.apply_all_transforms(
      x_train)
  x_val_transformed, transformations_inds_val = transformer.apply_all_transforms(
      x_val)
  mdl = DeepHits(input_shape=x_train.shape[1:],
                 n_classes=transformer.n_transforms)
  mdl.fit(x_train_transformed, to_categorical(transformations_inds),
          verbose=1,
          epochs=100, batch_size=128, validation_data=(
      x_val_transformed, to_categorical(transformations_inds_val)), patience=1)
  mdl.eval_tf(x_train_transformed, to_categorical(transformations_inds),
              verbose=1)
  mdl.eval_tf(x_val_transformed, to_categorical(transformations_inds_val),
              verbose=1)
  mdl.save_weights('dummy.ckpt')
  # print(mdl.layers)
  del mdl
  mdl = DeepHits(input_shape=x_train.shape[1:],
                 n_classes=transformer.n_transforms)
  # # mdl.build((None,) + x_train.shape[1:])
  # print(mdl.layers)
  # mdl.eval_tf(x_train_transformed, to_categorical(transformations_inds),
  #             verbose=1)
  mdl.load_weights('dummy.ckpt').expect_partial()
  mdl.eval_tf(x_train_transformed, to_categorical(transformations_inds),
              verbose=1)
  mdl.eval_tf(x_val_transformed, to_categorical(transformations_inds_val),
              verbose=1)