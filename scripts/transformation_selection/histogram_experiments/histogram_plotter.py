"""
Training a model with basic-non-composed transforms, to visualize it feature
 layer and
then calculate FID
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class HistogramPlotterResultDict(object):

  def __init__(self, results_path=None):
    self._results_folder_path = self._init_results_path(results_path)
    self._transformation_names_list = self._init_transformation_names_list()

  def _init_results_path(self, results_path):
    if results_path is None:
      results_folder_path = os.path.join(
          PROJECT_PATH, 'scripts', 'transformation_selection',
          'histogram_experiments', 'results')
      return results_folder_path
    return results_path

  def _init_transformation_names_list(self):
    trf_names = [
      'none', 'flip', 'rot_1', 'rot_2', 'rot_3', 'gauss', 'laplace', 'shiftX_8',
      'shiftX_-8', 'shiftY_8', 'shiftY_-8']
    return trf_names

  # def _truncate_laplace(self, results_list):
  #   laplace_idx = 6
  #   laplace_value = results_list[laplace_idx]
  #   results_array = np.array(results_list)
  #   results_array.sort()
  #   second_bigeest = results_array[-2]
  #   if laplace_value > second_bigeest*3

  def _get_dict_expname_results_grouped_by_regex(self, regex):
    dict_expname_result = {}
    result_names = [f for f in listdir(self._results_folder_path) if
                    isfile(join(self._results_folder_path, f))]
    result_names_given_dataloader = [name for name in result_names if
                                     regex in name]
    # print(result_names_given_dataloader)
    for result_file_name_i in result_names_given_dataloader:
      result_path_i = os.path.join(self._results_folder_path,
                                   result_file_name_i)
      result_name_i = result_file_name_i.split('.')[0]
      dict_expname_result[result_name_i] = pd.read_pickle(result_path_i)
    return dict_expname_result

  def _get_result_in_order_of_trf_names(self, results_dict: dict):
    results_list = []
    for trf_name in self._transformation_names_list:
      results_list.append(results_dict[trf_name])
    return results_list

  def _0_1_norm_list(self, list_to_norm):
    list_to_norm = np.array(list_to_norm)
    norm_list = list_to_norm - np.min(list_to_norm)
    norm_list = norm_list / np.max(norm_list)
    return norm_list

  def plot_histograms_by_regex(self, regex: str, log_scale=False):
    dict_expname_result = self. \
      _get_dict_expname_results_grouped_by_regex(regex)
    # print(list(dict_expname_result.keys()))
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for exp_name in dict_expname_result.keys():
      # print(exp_name)
      results_dict = dict_expname_result[exp_name]
      # print(results_dict)
      results_list = self._get_result_in_order_of_trf_names(results_dict)
      # print(results_list)
      aux_idxs = list(range(len(results_list)))
      norm_results = self._0_1_norm_list(results_list)
      ax.plot(aux_idxs, norm_results, label=exp_name)
    if log_scale:
      ax.set_yscale('log')
    ax.set_title('Transformation Distances Grouped By "%s"' % regex)
    # box = ax.get_position()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.xaxis.set_ticks(aux_idxs)
    ax.set_xticklabels(self._transformation_names_list)
    plt.show()


if __name__ == "__main__":
  hist_plotter = HistogramPlotterResultDict()
  hist_plotter.plot_histograms_by_regex('Rdm', log_scale=False)
