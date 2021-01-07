"""
HiTS stamps outlier loader

It loads pickles that are generated by thesis_hits.py

Data is already preprocessed; [-1,1] normed

loaded pickle are of format
data_tuples = (
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test))
"""

import os
import sys

import pandas as pd

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from parameters import loader_keys
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader


# TODO: refactor to integrate with ZTF dataset and
#  an easy coupling with other classic datasets
# Todo: Do some refactoring to include kwargs
class HiTSOutlierLoaderv2(HiTSOutlierLoader):

    def __init__(self, params: dict, dataset_name='hitsv2',
        pickles_usage=True):
        # super().__init__(params, dataset_name, pickles_usage)
        self.data_path = params.get(loader_keys.DATA_PATH, None)
        self.name = dataset_name# + '_%i_channels' % len(self.used_channels)

    def get_outlier_detection_datasets(self):
        return pd.read_pickle(self.data_path)