import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys

"""
See if chinos MI is compatible with TF2
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from modules.info_metrics.information_estimator_v2 import InformationEstimator
from modules.info_metrics.gaussian_dataset import generate_dataset

if __name__ == '__main__':

    dimension = 10
    n_samples_batch = 128
    sigma_zero = 2.0
    corr_factor = 0.9
    x_samples, y_samples, real_MI = generate_dataset(dimension, corr_factor,
                                                      n_samples_batch)

    # Estimation new rule
    estimator = InformationEstimator(sigma_zero, normalize_dimension=True)


    mi_estimation = estimator.mutual_information(x_samples, y_samples)

    print('Real MI: %.4f\nEstimated MI: %.4f' % (real_MI, mi_estimation))

