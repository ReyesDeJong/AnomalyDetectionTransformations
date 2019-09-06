"""
Generator for GAN

@author Esteban Reyes
"""

import tensorflow as tf
from parameters import param_keys, constants
from modules.networks.generator_tflib import GeneratorTflib
import tflib as lib
import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.ops.deconv2d


class GeneratorDCTflib(GeneratorTflib):
  """Generator neural network of a GAN."""

  def __init__(self, params, training_flag, n_samples_to_generate):
    super().__init__(params, training_flag, n_samples_to_generate)

  def _build_network(self, params, training_flag, n_samples_to_generate):
    with tf.variable_scope(constants.GENERATOR, reuse=tf.AUTO_REUSE):
      init_n_filters = params[param_keys.INITIAL_N_FILTERS]
      input_noise = self._get_input_noise(params, n_samples_to_generate)

      DIM = init_n_filters
      noise = input_noise
      MODE = 'dcgan'
      OUTPUT_DIM = 21 * 21

      output = lib.ops.linear.Linear('Generator.Input1', 128, 3 * 3 * 4 * DIM,
                                     noise)
      output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
      output = tf.nn.relu(output)

      output = tf.reshape(output, [-1, 4 * DIM, 3, 3])

      output = lib.ops.deconv2d.Deconv2D('Generator.2', 4 * DIM, 2 * DIM, 5,
                                         output)
      output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 2, 3], output)
      output = tf.nn.relu(output)

      output = lib.ops.deconv2d.Deconv2D('Generator.3', 2 * DIM, DIM, 5, output)
      output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 2, 3], output)
      output = tf.nn.relu(output)

      output = lib.ops.deconv2d.Deconv2D('Generator.4', DIM, 1, 5, output)
      output = tf.nn.sigmoid(output)

      output = output[:, :, :21, :21]

      output = tf.reshape(output, [-1, OUTPUT_DIM])

    return input_noise, output
