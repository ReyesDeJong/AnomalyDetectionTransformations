"""
Transformer object that performs 4 Rot; 4 Translations; Flip; Kernel Ops
without their composition
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname('__file__'), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform. \
  transformations_tf import AbstractTransformer, KernelTransformation

import numpy as np


class NoCompositionTransformer(AbstractTransformer):
  def __init__(self, translation_x=8, translation_y=8, rotations=True,
      flips=True, gauss=True, log=True, transform_batch_size=512,
      name='No_Composition_Transformer'):
    self.translation_x = translation_x
    self.translation_y = translation_y
    self.rotations = rotations
    self.flips = flips
    self.gauss = gauss
    self.log = log
    super().__init__(transform_batch_size, name)

  def _create_transformation_tuples_list(self):
    self.transformation_tuples = (
      (0, 0, 0, 0, 0, 0), (1 * self.flips, 0, 0, 0, 0, 0),
      (0, self.translation_x, 0, 0, 0, 0), (0, -self.translation_x, 0, 0, 0, 0),
      (0, 0, self.translation_y, 0, 0, 0), (0, 0, -self.translation_y, 0, 0, 0),
      (0, 0, 0, 1 * self.rotations, 0, 0), (0, 0, 0, 2 * self.rotations, 0, 0),
      (0, 0, 0, 3 * self.rotations, 0, 0),
      (0, 0, 0, 0, 1 * self.gauss, 0), (0, 0, 0, 0, 0, 1 * self.log)
    )
    # if some of the parameters is st to zero, avoid transformation redundance,
    # because original would appear more than once
    if self.translation_y * self.translation_x * self.rotations * \
        self.flips * self.gauss * self.log == 0:
      self.transformation_tuples = tuple(
          np.unique(self.transformation_tuples, axis=0))

  def _create_transformation_op_list(self):
    transformation_list = []
    for is_flip, tx, ty, k_rotate, is_gauss, is_log in self.transformation_tuples:
      transformation = KernelTransformation(is_flip, tx, ty, k_rotate, is_gauss,
                                            is_log)
      transformation_list.append(transformation)

    self._transformation_ops = transformation_list
