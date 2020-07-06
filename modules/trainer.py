"""
Train a model N times and return a specified metric on test set
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from parameters import general_keys, param_keys
import numpy as np
from typing import Callable
from models.transformer_od import TransformODModel


class ODTrainer(object):

    def __init__(self, input_params=dict()):
        params = self.get_default_parameters()
        params.update(input_params)
        self.score_name = params[param_keys.SCORE_NAME]
        self.metric_name_to_retrieve = params[param_keys.METRIC_NAME]
        self.train_epochs = params[param_keys.EPOCHS]

    def get_default_parameters(self):
        default_params = {
            param_keys.SCORE_NAME: general_keys.DIRICHLET,
            param_keys.METRIC_NAME: general_keys.ROC_AUC,
        }
        return default_params

    def train_and_evaluate_model_n_times(
        self, ModelClass: Callable[[], TransformODModel], transformer,
        x_train, x_val, x_test, y_test, train_times, verbose=False
    ):
        self.metric_results = []
        for i in range(train_times):
            model = ModelClass(
                None, transformer, input_shape=x_train.shape[1:])
            model.fit(x_train, x_val, epochs=self.train_epochs, patience=0,
                      verbose=verbose)
            results = model.evaluate_od(
                x_train, x_test, y_test, '', 'real', x_val, verbose=verbose)
            self.metric_results.append(results['dirichlet']['roc_auc'])
            del model
        # print(self.metric_results)
        return self.metric_results

    def print_metric_mean_and_std(self):
        msg = "%s : %.4f +/- %.4f" % \
               (self.metric_name_to_retrieve, np.mean(self.metric_results),
                np.mean(self.metric_results))
        print(msg)

    def get_metric_mean_and_std(self):
        return np.mean(self.metric_results), np.std(self.metric_results)

if __name__ == "__main__":
    from modules.geometric_transform.transformer_for_ranking import \
        RankingTransformer
    from models.transformer_od_simple_net import TransformODSimpleModel
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from parameters import loader_keys, general_keys
    import numpy as np

    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 10000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    data_loader = HiTSOutlierLoader(hits_params)

    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = data_loader.get_outlier_detection_datasets()

    trfer = RankingTransformer()
    model = TransformODSimpleModel(
        data_loader, trfer, input_shape=x_train.shape[1:])
    model.fit(x_train, x_val, epochs=1, patience=0)
    results = model.evaluate_od(
        x_train, x_test, y_test,
        '', 'real', x_val)
    print('AUROC using %s %.5f' % (
        data_loader.name,
        results['dirichlet']['roc_auc']))
