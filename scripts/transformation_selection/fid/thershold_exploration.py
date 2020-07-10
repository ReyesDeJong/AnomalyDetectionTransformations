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

from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.transformation_selection.histogram_experiments.histogram_plotter\
  import HistogramPlotterResultDict
from sklearn.cluster import KMeans

if __name__ == "__main__":
  hist_plotter = HistogramPlotterResultDict()
  fid_array = hist_plotter.get_result_name_values_lists(
      # 'RawFID_hits_4_channels',
      'RawFID_ztf_3_channels',
      exlude_list=['Rdm'])[1][0]
  log_fid = np.log(fid_array)
  print(fid_array)
  print(log_fid)
  print(hist_plotter._transformation_names_list)
  # plt.plot([1]*len(fid_array), fid_array, 'o')
  # plt.show()
  #
  # data = fid_array
  # data = data[... ,None]
  # kmeans = KMeans(n_clusters=2)
  # kmeans.fit(data)
  # y_kmeans = kmeans.predict(data)
  # plt.scatter([1]*len(data), data[:, 0], c=y_kmeans)
  # plt.show()


  data = log_fid[1:]
  plt.plot([1]*len(data), data, 'o')
  plt.show()
  data = data[... ,None]
  kmeans = KMeans(n_clusters=2)
  kmeans.fit(data)
  y_kmeans = kmeans.predict(data)
  plt.scatter([1]*len(data), data[:, 0], c=y_kmeans)
  plt.show()
