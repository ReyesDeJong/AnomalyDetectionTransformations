"""
Discriminator for GAN

@author Esteban Reyes
"""

import tensorflow as tf
from parameters import param_keys, constants
from modules.networks.discriminator import Discriminator
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm


class DiscriminatorTflib(Discriminator):
  """Discriminator neural network of a GAN."""

  def __init__(self, inputs, params, training_flag):
    super().__init__(inputs, params, training_flag)

  def LeakyReLU(self, x, alpha=0.2):
    return tf.maximum(alpha * x, x)

  def _build_network(self, inputs, params, training_flag):
    with tf.variable_scope(constants.DISCRIMINATOR, reuse=tf.AUTO_REUSE):
      init_n_filters = params[param_keys.INITIAL_N_FILTERS]

      DIM = init_n_filters
      MODE = 'wgan-gp'

      output = tf.reshape(inputs, [-1, 1, 21, 21])
      output = lib.ops.conv2d.Conv2D('Discriminator.1', 1, DIM, 5, output,
                                     stride=2)
      output = self.LeakyReLU(output)

      output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2 * DIM, 5, output,
                                     stride=2)

      if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3],
                                             output)

      output = self.LeakyReLU(output)
      output = lib.ops.conv2d.Conv2D('Discriminator.3', 2 * DIM, 4 * DIM, 5,
                                     output, stride=2)

      if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3],
                                             output)

      output = self.LeakyReLU(output)
      output = tf.reshape(output, [-1, 4 * 3 * 3 * DIM])
      output = lib.ops.linear.Linear('Discriminator.Output4', 4 * 3 * 3 * DIM, 1,
                                     output)

      output = tf.reshape(output, [-1])

    return output
