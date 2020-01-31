"""
By training an OVO ensemble, we see which transformations are distinguishable
from one another.
Distinguishability is characterized by accuracy between classes,
the idea is to return a new transformer object, which only contains the
distinguishable transformations

Tp see transformation set verbose to 1
"""

import os
import sys

import numpy as np
import tensorflow as tf

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.geometric_transform.transformations_tf import AbstractTransformer
from models.transformer_ensemble_ovo_od import EnsembleOVOTransformODModel


# Todo: do a checking between validation and train accuracy checking,
# because if model is overfitted they may return different results
class TransformSelectorByOVO(tf.keras.Model):

  def get_transformer_with_selected_transforms(self,
      model: EnsembleOVOTransformODModel, accuracy_matrix: np.ndarray,
      transformer: AbstractTransformer, selection_accuracy_tolerance=0.01,
      verbose=0):
    self.redundant_transforms_tuples = []
    for x_y_tuple in model.models_index_tuples:
      x_ind = x_y_tuple[0]
      y_ind = x_y_tuple[1]
      x_y_acc = accuracy_matrix[x_ind, y_ind]
      accuracy_interval = np.abs(x_y_acc - 0.5)
      if accuracy_interval <= selection_accuracy_tolerance:
        self.redundant_transforms_tuples.append(x_y_tuple)
    if verbose:
      print('Conflicting transformations')
      for conflicting_tuple in self.redundant_transforms_tuples:
        print('(%i,%i): %s ; %s' % (
          conflicting_tuple[0], conflicting_tuple[1],
          str(transformer.tranformation_to_perform[conflicting_tuple[0]]),
          str(transformer.tranformation_to_perform[conflicting_tuple[1]])))
    # TODO: do a random selection and, selection_accuracy_tolerance=accuracy_selection_tolerance) a most repeated based. THIS is first
    #  chosen
    transforms_to_delete = [x_y_tuple[1] for x_y_tuple in
                                 self.redundant_transforms_tuples]
    unique_transforms_to_delete = np.unique(transforms_to_delete)
    if verbose:
      print(transforms_to_delete)
      print(unique_transforms_to_delete)
    reversed_unique_transfors_to_delete = unique_transforms_to_delete[::-1]
    for i in reversed_unique_transfors_to_delete:
      del transformer.tranformation_to_perform[i]
      del transformer._transformation_list[i]
    if verbose:
      print(
        'Left Transformations %i' % len(transformer.tranformation_to_perform))
      print(transformer.tranformation_to_perform)
    return transformer
