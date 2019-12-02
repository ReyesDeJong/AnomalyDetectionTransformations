from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from parameters import constants

WEIGHT_DECAY = 0.5 * 0.0005


def _get_channels_axis(data_format):
  return -1 if data_format == 'channels_last' else 1


# TODO: data_format to constructors of layers
class WideResidualNetwork(tf.keras.Model):
  def __init__(self, input_shape, n_classes, depth,
      widen_factor=1, dropout_rate=0.0, final_activation='softmax',
      name='wide_resnet', data_format='channels_last',
      weight_decay=WEIGHT_DECAY):
    super().__init__(name=name)
    self.inp_shape = input_shape
    self.input_channels = input_shape[_get_channels_axis(data_format)]
    self.data_format = data_format
    self.weight_decay = weight_decay
    # used in kernels and batch_norm
    self.regularizer = tf.keras.regularizers.l2(self.weight_decay)
    self.conv_initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=constants.FAN_IN,
        distribution=constants.UNTRUNCATED_NORMAL)
    self.dense_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0, mode=constants.FAN_IN,
        distribution=constants.UNIFORM)

    n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
    assert ((depth - 4) % 6 == 0), 'depth should be 6n+4'
    n_residual_blocks = (depth - 4) // 6

    # self.input_layer = tf.keras.layers.InputLayer(input_shape)
    self.conv_1 = self.conv2d(n_channels[0], 3, 1)
    self.group_1 = self.conv_group(
        self.input_channels, n_channels[1], n_residual_blocks, 1, dropout_rate)
    self.group_2 = self.conv_group(
        self.input_channels, n_channels[2], n_residual_blocks, 2, dropout_rate)
    self.group_3 = self.conv_group(
        self.input_channels, n_channels[3], n_residual_blocks, 2, dropout_rate)
    self.bn_1 = self.batch_norm()
    self.act_1 = tf.keras.layers.Activation('relu')
    self.gap_1 = tf.keras.layers.GlobalAveragePooling2D()
    self.fc_1 = self.dense(n_classes)
    self.act_out = tf.keras.layers.Activation(final_activation)

  def call(self, input_tensor, training=False):
    # x = self.input_layer(input_tensor)
    x = self.conv_1(input_tensor)  # x)
    x = self.group_1(x, training=training)
    x = self.group_2(x, training=training)
    x = self.group_3(x, training=training)
    x = self.bn_1(x, training=training)
    x = self.act_1(x)
    x = self.gap_1(x)
    x = self.fc_1(x)
    x = self.act_out(x)
    return x

  def model(self):
    x = tf.keras.layers.Input(shape=self.inp_shape)
    return tf.keras.Model(inputs=x, outputs=self.call(x))

  def conv_group(self, in_channels, out_channels, n_res_blocks, strides,
      dropout_rate=0.0):
    blocks = []
    blocks.append(ResnetBlock(in_channels, out_channels, strides, dropout_rate))
    for _ in range(1, n_res_blocks):
      blocks.append(ResnetBlock(out_channels, out_channels, 1, dropout_rate))
    return tf.keras.Sequential(blocks)

  def get_output(self):
    return self.logits

  # TODO: move elsewhere to avoid code replication
  def conv2d(self, filters, kernel_size, strides):
    return tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=constants.SAME, use_bias=False,
        kernel_initializer=self.conv_initializer,
        kernel_regularizer=self.regularizer, data_format=self.data_format)

  # TODO: move elsewhere to avoid code replication
  def batch_norm(self):
    return tf.keras.layers.BatchNormalization(
        axis=_get_channels_axis(self.data_format), momentum=0.9, epsilon=1e-5,
        beta_regularizer=self.regularizer, gamma_regularizer=self.regularizer)

  # TODO: move elsewhere to avoid code replication
  def dense(self, units):
    return tf.keras.layers.Dense(
        units, kernel_initializer=self.dense_initializer,
        kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)


class ResnetBlock(tf.keras.Model):
  def __init__(self, in_channels, filters, strides=1,
      drop_rate=0, data_format='channels_last',
      name='resnet_block'):
    super().__init__(name=name)
    self.weight_decay = 0.5 * 0.0005
    self.kernel_size = 3
    self.data_format = data_format
    # used in kernels and batch_norm
    self.regularizer = tf.keras.regularizers.l2(self.weight_decay)
    self.kernel_initializer = tf.initializers.VarianceScaling(
        scale=2.0, mode=constants.FAN_IN,
        distribution=constants.UNTRUNCATED_NORMAL)

    self.bn_1 = self.batch_norm()
    self.act_1 = tf.keras.layers.Activation('relu')
    self.conv_1 = self.conv2d(filters, self.kernel_size, strides)
    self.bn_2 = self.batch_norm()
    self.act_2 = tf.keras.layers.Activation('relu')
    self.dropout_1 = tf.keras.layers.Dropout(drop_rate)
    self.conv_2 = self.conv2d(filters, self.kernel_size, 1)
    self.shorcut_1 = self.shortcut_layer(in_channels, filters, strides)

  def call(self, input_tensor, training=False):
    x = self.bn_1(input_tensor, training=training)
    act_1 = self.act_1(x)
    x = self.conv_1(act_1)
    x = self.bn_2(x, training=training)
    x = self.act_2(x)
    x = self.dropout_1(x, training=training)
    x = self.conv_2(x)
    shortcut = self.shorcut_1(act_1)
    x += shortcut
    return x

  def batch_norm(self):
    return tf.keras.layers.BatchNormalization(
        axis=_get_channels_axis(self.data_format), momentum=0.9, epsilon=1e-5,
        beta_regularizer=self.regularizer, gamma_regularizer=self.regularizer)

  def conv2d(self, filters, kernel_size, strides):
    return tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=constants.SAME, use_bias=False,
        kernel_initializer=self.kernel_initializer,
        kernel_regularizer=self.regularizer, data_format=self.data_format)

  def shortcut_layer(self, in_channels, filters, strides):
    input_channels = in_channels  # input_tensor.shape[_get_channels_axis(self.data_format)]
    is_input_output_channels_not_equal = input_channels != filters
    if is_input_output_channels_not_equal or strides != 1:
      return self.conv2d(filters, 1, strides)
    else:
      return tf.keras.layers.Activation(None)


if __name__ == '__main__':
  x_shape = (None, 21, 21, 3)
  n_transforms = 72
  depth, widen_factor = (10, 4)
  model = WideResidualNetwork(input_shape=x_shape[1:], n_classes=n_transforms,
                              depth=depth, widen_factor=widen_factor)
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  # model.build(x_shape)
  model.model().summary()

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
  #   x_train_task)
  # batch_size = 128
  #
  # mdl.fit(x=x_train_task_transformed,
  #         y=tf.keras.utils.to_categorical(transformations_inds),
  #         batch_size=batch_size,
  #         epochs=1  # int(np.ceil(200 / transformer.n_transforms))
  #         )
  # pred = mdl(x_test)
  # print(pred)
