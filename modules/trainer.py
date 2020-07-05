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
        self, ModelClass, transformer, x_train, x_val, x_test, y_test,
        train_times
    ):
        self.metric_results = []
        for i in range(train_times):
            model = ModelClass(
                None, transformer, input_shape=x_train.shape[1:])
            model.fit(x_train, x_val, epochs=self.train_epochs, patience=0)
            results = model.evaluate_od(
                x_train, x_test, y_test, '', 'real', x_val)
            self.metric_results.append(results['dirichlet']['roc_auc'])
            del model
        return self.metric_results

    def print_metric_mean_and_std(self):
        msg = "%s : %.4f +/- %.4f" % \
               (self.metric_name_to_retrieve, np.mean(self.metric_results),
                np.mean(self.metric_results))
        print(msg)

    def get_metric_mean_and_std(self):
        return np.mean(self.metric_results), np.mean(self.metric_results)
