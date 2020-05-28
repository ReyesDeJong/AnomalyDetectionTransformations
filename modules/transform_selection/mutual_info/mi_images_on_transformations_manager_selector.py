"""
Object containing MII for every transformation
It allows its visualization and transformation selection
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules import score_functions


class MIIOnTransformationsManager(object):
  def __init__(self, mii_dict: dict, transformer: AbstractTransformer):
    self.mii_dict = mii_dict
    self.transformer = transformer

  def _get_transformation_tuples_explain_txt(self, transformation_tuples_len):
    if (transformation_tuples_len) == 4:
      return '(Flip, Tx, Ty, Rot)'

    elif (transformation_tuples_len) == 6:
      return '(Flip, Tx, Ty, Rot, Gauss, LoG)'

  def get_mean_mii_dict(self) -> dict:
    mean_mii_dict = {}
    for key in self.mii_dict.keys():
      mean_mii_dict[key] = np.mean(self.mii_dict[key], axis=0)

    return mean_mii_dict

  def get_std_mii_dict(self) -> dict:
    std_mii_dict = {}
    for key in self.mii_dict.keys():
      std_mii_dict[key] = np.std(self.mii_dict[key], axis=0)

    return std_mii_dict

  def normalize_mean_mii_dict(self, mean_mii_dict):
    all_mean_mii_array = np.array(list(mean_mii_dict.values()))
    norm_mii_array = all_mean_mii_array
    norm_mii_array -= np.nanmin(all_mean_mii_array, axis=(1, 2))[:, np.newaxis,
                      np.newaxis]
    norm_mii_array = norm_mii_array / np.nanmax(norm_mii_array, axis=(1, 2))[
                                      :, np.newaxis, np.newaxis]
    norm_mii_dict = {}
    for i, key in enumerate(self.mii_dict.keys()):
      norm_mii_dict[key] = norm_mii_array[i]
    return norm_mii_dict

  def plot_mii_dict(self, plot_show=False, fig_size=20, norm_mii=True,
      extra_title_text=''):
    transformation_tuples = list(self.mii_dict.keys())
    transform_tuple_explain_text = self._get_transformation_tuples_explain_txt(
        len(transformation_tuples[0]))
    mean_mii_dict = self.get_mean_mii_dict()
    if norm_mii:
      mean_mii_dict = self.normalize_mean_mii_dict(mean_mii_dict)

    all_mean_mii_array = np.array(list(mean_mii_dict.values()))
    n_transformations = len(transformation_tuples)
    n_subplots_sqrt_side = int(np.ceil(np.sqrt(n_transformations)))
    fig, axs = plt.subplots(
        n_subplots_sqrt_side, n_subplots_sqrt_side,
        figsize=(fig_size, fig_size),
        gridspec_kw={'wspace': 0.01, 'hspace': 0.01}, constrained_layout=True)
    sup_title = transform_tuple_explain_text + '; norm mii: %s' % str(
        norm_mii) + '; %s' % extra_title_text
    fig.suptitle(sup_title, fontsize=fig_size * 2)
    axs = axs.flatten()
    vmin = np.min(all_mean_mii_array)
    vmax = np.max(all_mean_mii_array)
    for tuple_idx, tuple in enumerate(transformation_tuples):
      img = axs[tuple_idx].imshow(mean_mii_dict[tuple], vmax=vmax, vmin=vmin)
      mii_title = 'I(X;T(X)) %s' % (str(tuple))
      axs[tuple_idx].set_title(mii_title, fontsize=fig_size * 1.2)
      # if not norm_mii:
      #   divider = make_axes_locatable(axs[tuple_idx])
      #   cax = divider.append_axes('right', size='5%', pad=0.05)
      #   fig.colorbar(img, cax=cax, orientation='vertical')

    for ax in axs:
      ax.axis('off')

    if plot_show:
      plt.show()

    plt.close()

  def get_comparisons_with_auto_mii(self, mean_mii_dict=None,
      comparison_criteria='mse'):
    if mean_mii_dict is None:
      mean_mii_dict = self.get_mean_mii_dict()
    tuples = list(mean_mii_dict.keys())
    tuples_len = len(tuples[0])
    auto_mii_x_x_tuple = tuple([0] * tuples_len)
    auto_mii_x_x = mean_mii_dict[auto_mii_x_x_tuple]
    sorted_mii_by_criteria = {}
    for tuple_i in tuples:
      mii_i = mean_mii_dict[tuple_i]
      if comparison_criteria == 'mse':
        axis_where_to_perform_mean = tuple(np.arange(len(mii_i.shape)))[1:]
        comparison_value = np.mean(
            np.sqrt(np.sum(np.square(auto_mii_x_x - mii_i),
                           axis=axis_where_to_perform_mean)))

      elif comparison_criteria == 'entropy_diff':
        auto_mii_entropy = score_functions.get_entropy(auto_mii_x_x[None, ...])[
          0]
        mii_i_entropy = score_functions.get_entropy(mii_i[None, ...])[0]
        comparison_value = np.abs(auto_mii_entropy - mii_i_entropy)

      elif comparison_criteria == 'entropy':
        comparison_value = score_functions.get_entropy(mii_i[None, ...])[
          0]

      sorted_mii_by_criteria[tuple_i] = comparison_value
    return sorted_mii_by_criteria

  def plot_mii_dict_with_comp_criteria(self, plot_show=False, fig_size=20,
      norm_mii=True, extra_title_text='', criteria='mse'):
    transformation_tuples = list(self.mii_dict.keys())
    transform_tuple_explain_text = self._get_transformation_tuples_explain_txt(
        len(transformation_tuples[0]))
    mean_mii_dict = self.get_mean_mii_dict()
    if norm_mii:
      mean_mii_dict = self.normalize_mean_mii_dict(mean_mii_dict)

    comparison_criteria_dict = self.get_comparisons_with_auto_mii(mean_mii_dict,
                                                                  criteria)
    comp_tuples = np.array(list(comparison_criteria_dict.keys()))
    comp_values = list(comparison_criteria_dict.values())
    sorted_idx_comp_values = np.argsort(comp_values)
    sorted_array = comp_tuples[sorted_idx_comp_values]
    transformation_tuples = tuple([tuple(tuple_i) for tuple_i in list(sorted_array)])

    all_mean_mii_array = np.array(list(mean_mii_dict.values()))
    n_transformations = len(transformation_tuples)
    n_subplots_sqrt_side = int(np.ceil(np.sqrt(n_transformations)))
    fig, axs = plt.subplots(
        n_subplots_sqrt_side, n_subplots_sqrt_side,
        figsize=(fig_size, fig_size),
        gridspec_kw={'wspace': 0.01, 'hspace': 0.01}, constrained_layout=True)
    sup_title = transform_tuple_explain_text + \
                '\nsort criteria %s; norm mii: %s' % (criteria, str(
        norm_mii)) + '; %s' % extra_title_text
    fig.suptitle(sup_title, fontsize=fig_size * 2)
    axs = axs.flatten()
    vmin = np.min(all_mean_mii_array)
    vmax = np.max(all_mean_mii_array)
    for tuple_idx, tuple_i in enumerate(transformation_tuples):
      img = axs[tuple_idx].imshow(mean_mii_dict[tuple_i], vmax=vmax, vmin=vmin)
      mii_title = 'I(X;T(X)) %s\n%.4f' % (
        str(tuple_i), comparison_criteria_dict[tuple_i])
      axs[tuple_idx].set_title(mii_title, fontsize=fig_size * 1.2)
      # if not norm_mii:
      #   divider = make_axes_locatable(axs[tuple_idx])
      #   cax = divider.append_axes('right', size='5%', pad=0.05)
      #   fig.colorbar(img, cax=cax, orientation='vertical')

    for ax in axs:
      ax.axis('off')

    if plot_show:
      plt.show()

    plt.close()


if __name__ == '__main__':
  from modules.data_loaders.artificial_dataset_factory import \
    CirclesFactory
  from modules.transform_selection.mutual_info.mi_image_calculator import \
    MIImageCalculator
  from modules.info_metrics.information_estimator_by_batch import \
    InformationEstimatorByBatch
  from modules.geometric_transform.transformer_no_compositions import \
    NoCompositionTransformer

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  SHOW_PLOTS = True
  BATCH_SIZE = 512  # 2
  N_IMAGES = BATCH_SIZE * 4  # 7000  # BATCH_SIZE * 2
  WINDOW_SIZE = 3
  SIGMA_ZERO = 2.0

  circle_factory = CirclesFactory()
  mi_estimator = InformationEstimatorByBatch(SIGMA_ZERO, BATCH_SIZE)
  mi_image_calculator = MIImageCalculator(information_estimator=mi_estimator,
                                          window_size=WINDOW_SIZE)
  transformer = NoCompositionTransformer()
  images = circle_factory.get_final_dataset(N_IMAGES)
  mii_every_transform = mi_image_calculator.mii_for_transformations(
      images, transformer)
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=False)
  mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=True)
  # mii_every_transform = mi_image_calculator.mii_for_transformations(
  #     images, transformer, normalize_patches=True)
  # mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=False,
  #                                   extra_title_text='normed patches')
  # mii_every_transform.plot_mii_dict(plot_show=SHOW_PLOTS, norm_mii=True,
  #                                   extra_title_text='normed patches')
  print('')
