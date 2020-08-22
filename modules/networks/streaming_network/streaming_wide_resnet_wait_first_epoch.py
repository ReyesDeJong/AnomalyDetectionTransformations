from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.utils import delta_timer
import time
import numpy as np
from parameters import loader_keys, general_keys
from modules.networks.streaming_network.streaming_transformations_wide_resnet \
    import StreamingTransformationsWideResnet
from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
from modules.networks.train_step_tf2.wide_residual_network import ResnetBlock
from parameters import constants
from modules.print_manager import PrintManager

WEIGHT_DECAY = 0.5 * 0.0005

class StreamingWideResnetWait1Epoch(StreamingTransformationsWideResnet):

    def __init__(self, input_channels, transformer: AbstractTransformer,
        depth=10, widen_factor=4, drop_rate=0.0, weight_decay=WEIGHT_DECAY,
        final_activation='softmax', name='WRN_Streaming_Trfs_Wait1Epoch',
        results_folder_name=None):
        super().__init__(
            input_channels, transformer, depth, widen_factor, drop_rate,
            weight_decay, final_activation, name, results_folder_name)

    def _in_first_epoch_wait(self, wait_first_epoch, epoch):
        if not wait_first_epoch:
            return False
        else:
            return epoch == 0

    # TODO: implement some kind of train_loggin
    def fit(self, x_train, epochs, x_validation=None, batch_size=128,
        iterations_to_validate=None, patience=None, verbose=True,
        wait_first_epoch=False, log_file='train.log',
        iterations_to_print_train=None):
        validation_batch_size = 1024
        self._init_tensorboard_summaries()
        print_manager = PrintManager().verbose_printing(verbose)
        file = open(os.path.join(self.results_folder_path, log_file), 'w')
        print_manager.file_printing(file)
        print('\nTraining Initiated\n')
        self._initialize_training_attributes(x_train, batch_size)
        # if validation_data is None:
        #     return self._fit_without_validation(x, y, batch_size, epochs)
        assert patience is not None
        iterations_to_validate = self._set_validation_at_epochs_end_if_none(
            iterations_to_validate)
        if iterations_to_print_train is None:
            iterations_to_print_train = iterations_to_validate
        train_ds = self._get_training_dataset(x_train, batch_size)
        validation_ds = tf.data.Dataset.from_tensor_slices(
            (x_validation)).batch(validation_batch_size)
        for epoch in range(epochs):
            for transformation_index in range(self.transformer.n_transforms):
                for iteration_i, x_batch_train in enumerate(train_ds):
                    iteration = self._get_iteration_wrt_train_initialization(
                        iteration_i, epoch, transformation_index, batch_size,
                        x_train)
                    if iteration % iterations_to_validate == 0:
                        self._validate(
                            validation_ds, iteration, patience, epoch,
                            wait_first_epoch)
                        if self.check_early_stopping(patience) and not \
                            self._in_first_epoch_wait(wait_first_epoch, epoch):
                            print_manager.close()
                            return
                    images_transformed, transformation_indexes_oh = \
                        self._transform_train_batch_and_get_transform_indxs_oh(
                            x_batch_train)
                    step_loss, step_accuracy = self.train_step(
                        images_transformed, transformation_indexes_oh)
                    if iteration % iterations_to_print_train == 0:
                        self._print_at_train(
                            step_accuracy, step_loss, iteration, epoch)
        self.load_weights(
            self.best_model_weights_path)
        self._print_training_end()
        self._reset_metrics()
        print_manager.close()
        file.close()

    def _validate(
        self, validation_ds: tf.data.Dataset, iteration, patience, epoch,
        wait_first_epoch):
        for transformation_index in range(self.transformer.n_transforms):
            for x_val_batch in validation_ds:
                x_transformed, transformation_indexes_oh = self.\
                    _transform_evaluation_batch_and_get_transform_indexes_oh(
                    x_val_batch, transformation_index)
                self.eval_step(x_transformed, transformation_indexes_oh)
        # To correctily print patience, this must be the order of the following
        # 2 methods
        best_model_finding_message = self.check_best_model_save(
            iteration, epoch, wait_first_epoch)
        self._print_at_validate(iteration, patience, epoch,
                                best_model_finding_message)
        self._reset_metrics()

    def check_best_model_save(self, iteration, epoch, wait_first_epoch):
        self.best_model_so_far[
            general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] += 1
        output_message = ''
        if self._in_first_epoch_wait(wait_first_epoch, epoch):
            self.best_model_so_far[
                general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] = 0
        elif self.eval_loss.result() < self.best_model_so_far[general_keys.LOSS]:
            self.best_model_so_far[general_keys.LOSS] = self.eval_loss.result()
            self.best_model_so_far[
                general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] = 0
            self.best_model_so_far[general_keys.ITERATION] = iteration
            self.save_weights(self.best_model_weights_path)
            output_message = "\n\nNew best %s model: %s %.6f @ it %d\n" % (
                self.evaluation_set_name, general_keys.LOSS,
                self.best_model_so_far[general_keys.LOSS],
                self.best_model_so_far[general_keys.ITERATION])
        return output_message


if __name__ == '__main__':
    from modules.geometric_transform.\
        streaming_transformers.transformer_ranking import RankingTransformer
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from modules.utils import set_soft_gpu_memory_growth
    set_soft_gpu_memory_growth()

    EPOCHS = 1000
    ITERATIONS_TO_VALIDATE = 100  # 1000 # None
    ITERATIONS_TO_PRINT_TRAIN = 10
    PATIENCE = 1  # 0

    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 10000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [2],
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    data_loader = HiTSOutlierLoader(hits_params)
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = data_loader.get_outlier_detection_datasets()

    transformer = RankingTransformer()
    # transformer.set_transformations_to_perform(transformer.transformation_tuples*100)
    print(transformer.n_transforms)

    mdl = StreamingWideResnetWait1Epoch(x_train.shape[:-1], transformer)
    mdl.save_initial_weights(x_train, mdl.results_folder_path)
    mdl.fit(
        x_train, epochs=EPOCHS, x_validation=x_val, batch_size=128,
        patience=PATIENCE, iterations_to_print_train=ITERATIONS_TO_PRINT_TRAIN,
        iterations_to_validate=ITERATIONS_TO_VALIDATE, wait_first_epoch=True)
    mdl.evaluate(x_train)
    mdl.evaluate(x_val)
    # print('\nResults with random Initial Weights')
    # mdl.load_weights(os.path.join(mdl.results_folder_path, 'init.ckpt'))
    # mdl.evaluate(x_train)
    # mdl.evaluate(x_val)
    # mdl.fit(
    #     x_train, epochs=EPOCHS, x_validation=x_val, batch_size=128,
    #     patience=PATIENCE, iterations_to_validate=ITERATIONS_TO_VALIDATE)
    # mdl.evaluate(x_train)
    # mdl.evaluate(x_val)
    # results_folder_path = mdl.results_folder_path
    # print(os.path.abspath(results_folder_path))
    #
    # del mdl
    # mdl = StreamingTransformationsWideResnet(x_train.shape[:-1], transformer)
    # mdl.load_weights(os.path.join(results_folder_path, 'checkpoints',
    #                               'best_weights.ckpt'))
    # print('\nResults with model loaded')
    # # mdl.evaluate(x_train, batch_size=1000)
    # # mdl.evaluate(x_train, batch_size=1000)
    # mdl.evaluate(x_train)
    # mdl.evaluate(x_train)
    # mdl.evaluate(x_val)
    # mdl.evaluate(x_val)
    # # mdl.evaluate(x_val, batch_size=256)
