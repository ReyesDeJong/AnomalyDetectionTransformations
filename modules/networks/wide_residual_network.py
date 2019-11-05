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
  def __init__(self, input_shape, num_classes, depth,
      widen_factor=1, dropout_rate=0.0, final_activation='softmax',
      name='resnet_block', data_format='channels_last'):
    super().__init__(name=name)
    self.input_channels = input_shape[_get_channels_axis()]
    self.data_format = data_format
    # used in kernels and batch_norm
    self.regularizer = tf.keras.regularizers.l2(self.weight_decay)
    self.conv_initializer = tf.initializers.VarianceScaling(
        scale=2.0, mode=constants.FAN_IN,
        distribution=constants.UNTRUNCATED_NORMAL)
    self.dense_initializer = tf.initializers.VarianceScaling(
        scale=1.0, mode=constants.FAN_IN,
        distribution=constants.UNIFORM)

    n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
    assert ((depth - 4) % 6 == 0), 'depth should be 6n+4'
    n_residual_blocks = (depth - 4) // 6

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
    self.fc_1 = self.dense(num_classes)
    self.act_out = tf.keras.layers.Activation(final_activation)

  def call(self, input_tensor, training=False):
    x = self.conv_1(input_tensor)
    x = self.bn_2(x, training=training)
    x = self.act_2(x)
    x = self.dropout_1(x)
    x = self.conv_2(x)
    x = self.shorcut_1(x)
    x += input_tensor
    return x

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
      data_format='channels_last', drop_rate=0,
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
    x = self.act_1(x)
    x = self.conv_1(x)
    x = self.bn_2(x, training=training)
    x = self.act_2(x)
    x = self.dropout_1(x, training=training)
    x = self.conv_2(x)
    x = self.shorcut_1(x)
    x += input_tensor
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
    is_input_output_channels_equal = input_channels == filters
    if is_input_output_channels_equal or strides != 1:
      return tf.keras.layers.Activation(None)
    else:
      return self.conv2d(filters, 1, strides)
