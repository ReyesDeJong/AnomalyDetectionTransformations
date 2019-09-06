"""
Discriminator for GAN

@author Esteban Reyes
"""

import tensorflow as tf
from modules import layers
from parameters import param_keys, constants


class Discriminator(object):
  """Discriminator neural network of a GAN."""

  def __init__(self, inputs, params, training_flag):
    # ToDo: define attributes
    """Constructor that builds network and output the logits

    Parameters
    ----------
    inputs : tf.Tensor
      Input to discriminator, i.e. fake or real images.
    params : dict
      Dictionary with model and network general parameters.
    training_flag : tf.Tensor
      Placeholder of a boolean tensor which indicates if it is the training
      phase or not.
    """
    self.output = self._build_network(inputs, params, training_flag)

  # ToDo: docstring
  # ToDo: scope in discriminator shouldn't be necessary
  # ToDo: summary to dense output when its activation is None
  def _build_network(self, inputs, params, training_flag):
    with tf.variable_scope(constants.DISCRIMINATOR, reuse=tf.AUTO_REUSE):
      # some parameters
      input_channels = params[param_keys.N_INPUT_CHANNELS]
      input_image_size = params[param_keys.INPUT_IMAGE_SIZE]
      kenel_size = params[param_keys.KERNEL_SIZE]
      init_n_filters = params[param_keys.INITIAL_N_FILTERS]
      batchnorm = params[param_keys.BATCHNORM_CONV]
      # kernel initializers
      dense_initializer = tf.contrib.layers.variance_scaling_initializer(
          factor=1.0, mode=constants.FAN_AVG,
          uniform=True)  # xavier or glorot init
      conv_initializer = tf.contrib.layers.variance_scaling_initializer(
          factor=2.0, mode=constants.FAN_AVG,
          uniform=True)  # 'he' (not really) init
      # network
      vec2img = tf.reshape(inputs, [-1, input_image_size, input_image_size,
                                    input_channels], name='vec2img_1')
      # 21
      conv_1 = layers.conv2d(
          inputs=vec2img, filters=init_n_filters, training=training_flag,
          batchnorm=None, kernel_size=kenel_size, strides=2,
          activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer,
          name='conv_1_2')
      # 11
      conv_2 = layers.conv2d(
          inputs=conv_1, filters=init_n_filters * 2, training=training_flag,
          batchnorm=batchnorm, kernel_size=kenel_size, strides=2,
          activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer,
          name='conv_2_3')
      # 6
      conv_3 = layers.conv2d(
          inputs=conv_2, filters=init_n_filters * 4, training=training_flag,
          batchnorm=batchnorm, kernel_size=kenel_size, strides=2,
          activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer,
          name='conv_3_4')
      # 3
      flatten = tf.layers.flatten(conv_3, name='flatten_5')

      dense_1 = tf.layers.dense(
          inputs=flatten, units=1, activation=None,
          kernel_initializer=dense_initializer, name="dense_1_6")

      output = tf.reshape(dense_1, [-1], name='output_7')

    return output

  # ToDo: docstring
  def get_output(self):
    return self.output
