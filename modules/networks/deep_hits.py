from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from parameters import constants


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




if __name__ == '__main__':
  x_shape = (None, 21, 21, 3)
  n_transforms = 72
  model = DeepHits(input_shape=x_shape[1:], n_classes=n_transforms)
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  print(model.model().summary())
  #
  # from modules.data_loaders.base_line_loaders import load_ztf_real_bog
  # from modules.geometric_transform.transformations_tf import Transformer
  #
  # gpus = tf.config.experimental.list_physical_devices('GPU')
  # for gpu in gpus:
  #   tf.config.experimental.set_memory_growth(gpu, True)
  #
  # single_class_ind = 1
  # (x_train, y_train), (x_test, y_test) = load_ztf_real_bog()
  # transformer = Transformer(8, 8)
  # n, k = (10, 4)
  #
  # mdl = WideResidualNetwork(x_train.shape[1:], transformer.n_transforms, n, k)
  # mdl.compile('adam', 'categorical_crossentropy', ['acc'])
  #
  # x_train_task = x_train[y_train.flatten() == single_class_ind]
  #
  # x_train_task_transformed, transformations_inds = transformer.apply_all_transforms(
  #     x_train_task)
  # batch_size = 128
  #
  # mdl.fit(x=x_train_task_transformed,
  #         y=tf.train_step_tf2.utils.to_categorical(transformations_inds),
  #         batch_size=batch_size,
  #         epochs=1  # int(np.ceil(200 / transformer.n_transforms))
  #         )
  # pred = mdl(x_test)
  # print(pred)
