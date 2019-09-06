"""
Generator for GAN

@author Esteban Reyes
"""

import tensorflow as tf
from modules import layers
from parameters import param_keys, constants


class Generator(object):
  """Generator neural network of a GAN."""

  def __init__(self, params, training_flag, n_samples_to_generate):
    # ToDo: define attributes
    """Constructor that builds network and output the logits

    Parameters
    ----------
    params : dict
      Dictionary with model and network general parameters.
    training_flag : tf.Tensor
      Placeholder of a boolean tensor which indicates if it is the training
      phase or not.
    n_samples_to_generate : tf.Tensor
      Placeholder of a int tensor which indicates the noise dimension, thus how
      many samples will be generated.
    """
    self.input_noise, self.fake_data = self._build_network(params, training_flag,
                                                           n_samples_to_generate)

  def _get_input_noise(self, params, n_samples_to_generate):
    """Tensor of noises (Z sampled for GAN) to generate from

    The noise tensor is sampled from N(0,I), and has shape,
    [n_samples_to_generate, params[general_params.NOISE_DIM], meaning that the
    generator will generate n_samples_to_generate from the noise tensor.

    Parameters
    ----------
    params : dict
      Dictionary with model and network general parameters.
    n_samples_to_generate : tf.Tensor
      Placeholder of a int tensor which indicates the noise dimension, thus how
      many samples will be generated.

    Returns
    -------
    noise : tf.Tensor
      noise tensor sampled from N(0,I).
    """
    noise_dim = params[param_keys.NOISE_DIM]
    noise = tf.random_normal([n_samples_to_generate, noise_dim])
    return noise

  # ToDo: docstring
  # ToDo: scope in generator shouldn't be necessary
  def _build_network(self, params, training_flag, n_samples_to_generate):
    with tf.variable_scope(constants.GENERATOR, reuse=tf.AUTO_REUSE):
      # some parameters
      kenel_size = params[param_keys.KERNEL_SIZE]
      init_n_filters = params[param_keys.INITIAL_N_FILTERS]
      drop_rate = params[param_keys.DROP_RATE]
      batchnorm = params[param_keys.BATCHNORM_CONV]
      input_noise = self._get_input_noise(params, n_samples_to_generate)
      # kernel initializers
      dense_initializer = tf.contrib.layers.variance_scaling_initializer(
          factor=1.0, mode=constants.FAN_AVG,
          uniform=True)  # xavier or glorot init
      deconv_initializer = tf.contrib.layers.variance_scaling_initializer(
          factor=2.0, mode=constants.FAN_AVG,
          uniform=True)  # 'he' (not really) init
      #network
      dense_1 = layers.dense(
          inputs=input_noise, units=3 * 3 * 4 * init_n_filters,
          training=training_flag, batchnorm=batchnorm, drop_rate=drop_rate,
          activation=tf.nn.relu, kernel_initializer=dense_initializer,
          name='dense_1')

      vec2img = tf.reshape(dense_1, [-1, 3, 3, 4 * init_n_filters],
                           name='vec2img_2')
      # 6
      deconv_1 = layers.deconv2d(
          inputs=vec2img, filters=2 * init_n_filters, training=training_flag,
          batchnorm=batchnorm, kernel_size=kenel_size, activation=tf.nn.relu,
          kernel_initializer=deconv_initializer, name='deconv_1_3')
      # 12
      deconv_2 = layers.deconv2d(
          inputs=deconv_1, filters=init_n_filters, training=training_flag,
          batchnorm=batchnorm, kernel_size=kenel_size, activation=tf.nn.relu,
          kernel_initializer=deconv_initializer, name='deconv_2_4')
      # 24
      deconv_3 = layers.deconv2d(
          inputs=deconv_2, filters=1, training=training_flag,
          batchnorm=None, kernel_size=kenel_size, activation=tf.nn.sigmoid,
          kernel_initializer=deconv_initializer, name='deconv_3_5')

      image_crop = deconv_3[:, :21, :21, :]

      #output = tf.layers.flatten(image_crop, name='flatten_7')
      output = image_crop

    return input_noise, output

  # ToDo: docstring
  def get_output(self):
    return self.fake_data

  def get_input_noise(self):
    return self.input_noise
