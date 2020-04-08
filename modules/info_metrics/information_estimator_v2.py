"""
@author Nicolas Tapia, Daniel Baeza
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class InformationEstimator(object):
    """ TensorFlow implementation of estimator proposed in:

    Giraldo, L. G. S., Rao, M., & Principe, J. C. (2015). Measures of entropy
    from data using infinitely divisible kernels. IEEE Transactions on
    Information Theory, 61(1), 535-548.

    with modifications made by the authors to attenuate scale and dimension
    related artifacts. The originally proposed kernel size formula is:

    sigma = sigma_0 * n ^ (-1 / (4+d))

    If normalize_scale is set to True, then the variable is first normalized
    to zero mean and unit variance, as:

    x -> (x - mean) / sqrt(var + epsilon)

    which is equivalent to add the standard deviation as a multiplicative
    dependence in sigma, as done in the classical Silverman rule. This is done
    to achieve invariance to changes of scale during the mutual information
    estimation process. Epsilon is a small number to avoid division by zero.

    If normalize_dimension is set to True, then sigma is computed as:

    sigma = sigma_0 * sqrt(d) * n ^ (-1 / (4+d))

    This is done to center the distribution of pair-wise distances to the same
    mean across variables with different dimensions, and as a consequence
    to attenuate dimension related artifacts.

    Note that normalize_scale=False and normalize_dimension=False will give you
    the original version of the estimator.

    """

    def __init__(
            self,
            sigma_zero,
            alpha=1.01,
            epsilon=1e-8,
            normalize_scale=True,
            normalize_dimension=True,
            log_base=2,
    ):
        self.sigma_zero = sigma_zero
        self.alpha = alpha
        self.epsilon = epsilon
        self.normalize_scale = normalize_scale
        self.normalize_dimension = normalize_dimension
        self.log_base = log_base
        self.flatten_layer = tf.keras.layers.Flatten()

    def normalized_gram(self, x, sigma_x=None, x_is_image=False):
        """If sigma_x is provided, then that value will be used. Otherwise
        it will be automatically computed using the formula.
        If x_is_image is True, then the normalization of scale (if applicable)
        is done mixing dimensions. If false, each dimension is normalized
        independently.
        """
        if sigma_x is None:
            sigma_x = self._compute_sigma(x)
        if self.normalize_scale:
            x = self._normalize_variable(x, x_is_image)
        # Compute differences directly to avoid numerical instability
        pairwise_difference = x[:, tf.newaxis, :] - x[tf.newaxis, :, :]
        pairwise_squared_difference = tf.square(pairwise_difference)
        pairwise_distance = tf.reduce_sum(
            pairwise_squared_difference, axis=2)
        # Avoids small negatives due to numerical precision
        pairwise_distance = tf.nn.relu(pairwise_distance)
        # We don't bother with the normalization constant of the kernel
        # since it will be canceled during normalization of the Gram matrix
        den = 2 * (sigma_x ** 2)
        gram = tf.exp(-pairwise_distance / den)
        # Normalize gram
        x_dims = tf.shape(x)
        n = tf.dtypes.cast(x_dims[0], tf.float32)
        norm_gram = gram / n
        return norm_gram

    def entropy(self, x, sigma_x=None, x_is_image=False):
        """See 'normalized_gram' doc."""
        norm_gram = self.normalized_gram(x, sigma_x, x_is_image)
        entropy = self.entropy_with_gram(norm_gram)
        return entropy

    def joint_entropy(self, x, y, sigma_x=None, sigma_y=None,
                      x_is_image=False, y_is_image=False):
        """See 'normalized_gram' doc."""
        norm_gram_a = self.normalized_gram(x, sigma_x, x_is_image)
        norm_gram_b = self.normalized_gram(y, sigma_y, y_is_image)
        joint_entropy = self.joint_entropy_with_gram(norm_gram_a, norm_gram_b)
        return joint_entropy

    def mutual_information(self, x, y, sigma_x=None, sigma_y=None,
                           x_is_image=False, y_is_image=False):
        """See 'normalized_gram' doc."""
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        x = self.flatten_layer(x)
        y = self.flatten_layer(y)
        norm_gram_a = self.normalized_gram(x, sigma_x, x_is_image)
        norm_gram_b = self.normalized_gram(y, sigma_y, y_is_image)
        mi_xy = self.mutual_information_with_gram(norm_gram_a, norm_gram_b)
        return mi_xy

    def entropy_with_gram(self, norm_gram):
        # eigvals, _ = tf.linalg.eigh(norm_gram)
        eigvals, _ = tf.compat.v1.self_adjoint_eig(norm_gram)
        # try:
        #     eigvals, _ = tf.linalg.eigh(norm_gram)
        # except Exception as e: print(e)
        #     print('error')
        eigvals = tf.nn.relu(eigvals)  # Avoids small negatives

        # Ensure eigenvalues sum 1,
        # in case a numerical instability occurred.
        eigvals = eigvals / tf.reduce_sum(eigvals)

        sum_term = tf.reduce_sum(eigvals ** self.alpha)
        entropy = tf.math.log(sum_term) / (1.0 - self.alpha)
        entropy = entropy / np.log(self.log_base)
        return entropy

    def joint_entropy_with_gram(self, norm_gram_a, norm_gram_b):
        n = tf.dtypes.cast(tf.shape(norm_gram_a)[0], tf.float32)
        norm_gram = n * tf.multiply(norm_gram_a, norm_gram_b)
        joint_entropy = self.entropy_with_gram(norm_gram)
        return joint_entropy

    def mutual_information_with_gram(self, norm_gram_a, norm_gram_b):
        h_x = self.entropy_with_gram(norm_gram_a)
        h_y = self.entropy_with_gram(norm_gram_b)
        h_xy = self.joint_entropy_with_gram(norm_gram_a, norm_gram_b)
        mi_xy = h_x + h_y - h_xy
        return mi_xy

    def _compute_sigma(self, x):
        x_dims = tf.shape(x)
        n = tf.dtypes.cast(x_dims[0], tf.float32)
        d = tf.dtypes.cast(x_dims[1], tf.float32)
        sigma = self.sigma_zero * n ** (-1 / (4 + d))
        if self.normalize_dimension:
            sigma = sigma * tf.sqrt(d)
        return sigma

    def _normalize_variable(self, x, x_is_image):
        if x_is_image:
            mean_x = tf.reduce_mean(x)
            var_x = tf.reduce_mean(tf.square(x - mean_x))
        else:
            mean_x, var_x = tf.nn.moments(x, [0])
        std_x = tf.sqrt(var_x + self.epsilon)
        x = (x - mean_x) / std_x
        return x
