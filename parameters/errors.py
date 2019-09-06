"""Module that defines common errors for parameter values."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from . import constants


def check_valid_value(value, name, valid_list):
    """Raises a ValueError exception if value not in valid_list"""
    if value not in valid_list:
        msg = constants.ERROR_INVALID % (valid_list, name, value)
        raise ValueError(msg)


def check_nan_metric(metric, metric_mean):
  """Raises a ValueError exception if metric value is nan"""
  if np.isnan(metric_mean):
    msg = constants.ERROR_NAN_METRIC % (metric)
    raise ValueError(msg)
