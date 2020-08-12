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
from modules.networks.streaming_network.refactored_deep_hits import DeepHits
from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
from modules.print_manager import PrintManager

# TODO: manage weights saved in a better manner: save_path, non saving when not given, etc
class StreamingTransformationsDeepHits(DeepHits):

    def __init__(
        self, transformer: AbstractTransformer, drop_rate=0.5,
        final_activation='softmax', name='DH_Streaming_Trfs',
        results_folder_name=None):
        super().__init__(transformer.n_transforms, drop_rate,
                         final_activation, name, results_folder_name)
        self.transformer = transformer

    # # TODO: Not implemented
    # def _fit_without_validation(self, x, y, batch_size, epochs):
    #     self.evaluation_set_name = 'train'
    #     train_ds = tf.data.Dataset.from_tensor_slices(
    #         (x, y)).shuffle(10000).batch(batch_size, drop_remainder=True)
    #     for epoch in range(epochs):
    #         epoch_start_time = time.time()
    #         for it_i, (images, labels) in enumerate(train_ds):
    #             self.train_step(images, labels)
    #             # self.eval_step(images, labels)
    #         template = 'Epoch {}, Loss: {}, Acc: {}, Time: {}'
    #         print(template.format(epoch + 1,
    #                               self.eval_loss.result(),
    #                               self.eval_accuracy.result() * 100,
    #                               delta_timer(
    #                                   time.time() - epoch_start_time)
    #                               ))
    #         self.check_best_model_save(
    #             it_i + ((epoch + 1) * self.n_iterations_in_epoch))
    #         self.eval_loss.reset_states()
    #         self.eval_accuracy.reset_states()
    #     print('Total Training Time: {}'.format(
    #         delta_timer(time.time() - self.training_star_time)))

    def _get_training_dataset(self, x_train, batch_size):
        train_ds = tf.data.Dataset.from_tensor_slices((x_train)).\
            shuffle(10000).batch(batch_size, drop_remainder=True)
        return train_ds

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.call(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    # this method cannot use tf.function because of conflict with getting
    # transformation_ops from a list by tensor indixes, thus it is SLOW!
    def _transform_train_batch_and_get_transform_indxs_oh(
        self, x_train_batch):
        x_transformed, transformation_indexes = \
            self.transformer.transform_batch_with_random_indexes(
                x_train_batch)
        transformation_indexes_oh = tf.one_hot(
            transformation_indexes, depth=self.transformer.n_transforms)
        return x_transformed, transformation_indexes_oh

    def _get_iteration_wrt_train_initialization(
        self, iteration, epoch, transformation_i, batch_size, x_train):
        iteration_real = iteration + (epoch * self.n_iterations_in_epoch) + (
            transformation_i * (len(x_train) // batch_size))
        return iteration_real

    def _initialize_training_attributes(self, x_data, batch_size):
        self.best_model_so_far = {
            general_keys.ITERATION: 0,
            general_keys.LOSS: 1e100,
            general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH: 0,
        }
        self.training_star_time = time.time()
        self.n_iterations_in_epoch = (
            len(x_data) // batch_size) * self.transformer.n_transforms
        self.evaluation_set_name = 'validation'

    def _set_validation_at_epochs_end_if_none(self, iterations_to_validate):
        if iterations_to_validate is None or iterations_to_validate==0:
            # -1 is used to actually perform a validation when epochs set to 1
            iterations_to_validate = self.n_iterations_in_epoch #- 1
        return iterations_to_validate

    # TODO: implement some kind of train_loggin
    def fit(self, x_train, epochs, x_validation=None, batch_size=128,
        iterations_to_validate=None, patience=None, verbose=True):
        validation_batch_size = 1024
        print_manager = PrintManager().verbose_printing(verbose)
        print('\nTraining Initiated\n')
        self._initialize_training_attributes(x_train, batch_size)
        # if validation_data is None:
        #     return self._fit_without_validation(x, y, batch_size, epochs)
        assert patience is not None
        iterations_to_validate = self._set_validation_at_epochs_end_if_none(
            iterations_to_validate)
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
                            validation_ds, iteration, patience, epoch)
                        if self.check_early_stopping(patience):
                            print_manager.close()
                            return
                    images_transformed, transformation_indexes_oh = \
                        self._transform_train_batch_and_get_transform_indxs_oh(
                            x_batch_train)
                    self.train_step(
                        images_transformed, transformation_indexes_oh)
        self.load_weights(
            self.best_model_weights_path)
        self._print_training_end()
        self._reset_metrics()
        print_manager.close()

    def _transform_evaluation_batch_and_get_transform_indexes_oh(
        self, x_batch_val, transform_index):
        transformation_indexes = tf.ones(
            x_batch_val.shape[0], dtype=tf.int32) * transform_index
        x_transformed = self.transformer.apply_specific_transform_on_batch(
                x_batch_val, transform_index)
        transformation_indexes_oh = tf.one_hot(
            transformation_indexes, depth=self.transformer.n_transforms)
        return x_transformed, transformation_indexes_oh

    def _print_at_validate(self, iteration, patience, epoch,
        best_model_missing_message):
        template = 'Time usage: %s\nEpoch %i Iteration %i Patience left %i' \
                   ' (train): loss %.6f, acc %.6f\n' \
                   '(validation): loss %.6f, acc %.6f %s'
        print(template % (
                delta_timer(time.time() - self.training_star_time),
                # (iteration!=0)+1 is added so in first val epochs = 0 and +1
                # in rest
                epoch, #+ (iteration!=0)*1,
                iteration,
                patience - self.best_model_so_far[
                    general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH],
                self.train_loss.result(),
                self.train_accuracy.result(),
                self.eval_loss.result(),
                self.eval_accuracy.result(),
                best_model_missing_message
            )
        )

    def _validate(
        self, validation_ds: tf.data.Dataset, iteration, patience, epoch):
        for transformation_index in range(self.transformer.n_transforms):
            for x_val_batch in validation_ds:
                x_transformed, transformation_indexes_oh = self.\
                    _transform_evaluation_batch_and_get_transform_indexes_oh(
                    x_val_batch, transformation_index)
                self.eval_step(x_transformed, transformation_indexes_oh)
        # To correctily print patience, this must be the order of the following
        # 2 methods
        best_model_finding_message = self.check_best_model_save(iteration)
        self._print_at_validate(iteration, patience, epoch,
                                best_model_finding_message)
        self._reset_metrics()

    def check_best_model_save(self, iteration):
        self.best_model_so_far[
            general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] += 1
        output_message = ''
        if self.eval_loss.result() < self.best_model_so_far[general_keys.LOSS]:
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

    def predict(self, x, batch_size=1024):
        eval_ds = tf.data.Dataset.from_tensor_slices((x)).batch(batch_size)
        predictions = []
        for images in eval_ds:
            predictions.append(self.call_wrapper_to_predict(images))
        return np.concatenate(predictions, axis=0)

    def _get_set_name_if_none(self, set_name):
        if set_name is None:
            return ''
        else:
            return '(%s) ' % set_name

    def evaluate(self, x_data, batch_size=1024, verbose=True, set_name=None):
        print_manager = PrintManager().verbose_printing(verbose)
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_data)).batch(batch_size)
        start_time = time.time()
        set_name = self._get_set_name_if_none(set_name)
        for transformation_index in range(self.transformer.n_transforms):
            for x_eval_batch in dataset:
                x_transformed, transformation_indexes_oh = self. \
                    _transform_evaluation_batch_and_get_transform_indexes_oh(
                    x_eval_batch, transformation_index)
                self.eval_step(x_transformed, transformation_indexes_oh)
        message = '%sloss %.6f, acc %.6f, time %s' % (
            set_name, self.eval_loss.result(), self.eval_accuracy.result(),
            delta_timer(time.time() - start_time))
        print(message)
        # print('')
        results_dict = {general_keys.LOSS: self.eval_loss.result(),
                        general_keys.ACCURACY: self.eval_accuracy.result()}
        self._reset_metrics()
        print_manager.close()
        return results_dict


if __name__ == '__main__':
    from modules.geometric_transform.streaming_transformers.transformer_ranking import \
        RankingTransformer
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from modules.utils import set_soft_gpu_memory_growth
    set_soft_gpu_memory_growth()

    EPOCHS = 1000
    ITERATIONS_TO_VALIDATE = 10  # 1000 # None
    PATIENCE = 0  # 0

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

    mdl = StreamingTransformationsDeepHits(transformer)
    mdl.save_initial_weights(x_train, mdl.results_folder_path)
    mdl.fit(
        x_train, epochs=EPOCHS, x_validation=x_val, batch_size=128,
        patience=PATIENCE, iterations_to_validate=ITERATIONS_TO_VALIDATE)
    mdl.evaluate(x_train)
    mdl.evaluate(x_val)
    print('\nResults with random Initial Weights')
    mdl.load_weights(os.path.join(mdl.results_folder_path, 'init.ckpt'))
    mdl.evaluate(x_train)
    mdl.evaluate(x_val)
    mdl.fit(
        x_train, epochs=EPOCHS, x_validation=x_val, batch_size=128,
        patience=PATIENCE, iterations_to_validate=ITERATIONS_TO_VALIDATE)
    mdl.evaluate(x_train)
    mdl.evaluate(x_val)

    del mdl
    mdl = StreamingTransformationsDeepHits(transformer)
    mdl.load_weights(os.path.join(mdl.results_folder_path, 'checkpoints',
                                  'best_weights.ckpt'))
    print('\nResults with model loaded')
    # mdl.evaluate(x_train, batch_size=1000)
    # mdl.evaluate(x_train, batch_size=1000)
    mdl.evaluate(x_train)
    mdl.evaluate(x_train)
    mdl.evaluate(x_val)
    mdl.evaluate(x_val)
    # mdl.evaluate(x_val, batch_size=256)
