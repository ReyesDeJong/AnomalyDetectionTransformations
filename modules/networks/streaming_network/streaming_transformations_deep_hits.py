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


# TODO: manage weights saved in a better manner: save_path, non saving when not given, etc
class StreamingTransformationsDeepHits(DeepHits):

    def __init__(
        self, transformer: AbstractTransformer, drop_rate=0.5,
        final_activation='softmax', name='deep_hits_streaming_transformations',
        results_folder_name=''):
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
    def _transform_batch_and_get_transformation_indexes_oh(self, x_data):
        x_transformed, transformation_indexes = \
            self.transformer.transform_batch_with_random_indexes(
                x_data)
        transformation_indexes_oh = tf.one_hot(
            transformation_indexes,
            depth=self.transformer.n_transforms)
        return x_transformed, transformation_indexes_oh

    def _get_iteration_wrt_train_initialization(
        self, iteration, epoch, transformation_i, batch_size, x_train):
        iteration_real = iteration + (epoch * self.n_iterations_in_epoch) + (
            transformation_i * (len(x_train) // batch_size))
        return iteration_real

    # TODO: implement some kind of train_loggin
    def fit(self, x, epochs, x_validation=None, batch_size=128,
        iterations_to_validate=None, patience=None, verbose=True):
        validation_batch_size = 1024
        self.print_manager.verbose_printing(verbose)
        print('\nTraining Initiated')
        self.training_star_time = time.time()
        self.best_model_so_far = {
            general_keys.ITERATION: 0,
            general_keys.LOSS: 1e100,
            general_keys.NOT_IMPROVED_COUNTER: 0,
        }
        self.n_iterations_in_epoch = (
            len(x) // batch_size) * self.transformer.n_transforms
        # if validation_data is None:
        #     return self._fit_without_validation(x, y, batch_size, epochs)
        assert patience is not None
        self.evaluation_set_name = 'validation'
        # check if validate at end of epoch
        if iterations_to_validate is None:
            # -1 comes from fact that iterations start at 0
            iterations_to_validate = self.n_iterations_in_epoch - 1

        train_ds = self._get_training_dataset(x, batch_size)
        validation_ds = tf.data.Dataset.from_tensor_slices(
            (x_validation)).batch(validation_batch_size)
        for epoch in range(epochs):
            for transformation_index in range(self.transformer.n_transforms):
                for iteration_i, x_images in enumerate(train_ds):
                    images_transformed, transformation_indexes_oh = \
                        self._transform_batch_and_get_transformation_indexes_oh(
                            x_images)
                    self.train_step(
                        images_transformed, transformation_indexes_oh)
                    iteration_i = self._get_iteration_wrt_train_initialization(
                        iteration_i, epoch, transformation_index, batch_size, x)
                    print(iteration_i, epoch, transformation_index, iterations_to_validate, self.transformer.n_transforms, len(x) // batch_size)
                    # TODO: slow
                    if iteration_i % iterations_to_validate == 0:
                        # if self.check_early_stopping(patience):
                        #     return
                        for transformation_idx_i in range(
                            self.transformer.n_transforms):
                            for validation_images in \
                                validation_ds:
                                transformations_inds = \
                                    [transformation_idx_i] * \
                                    validation_images.shape[0]
                                x_transformed = \
                                    self.transformer.apply_specific_transform(
                                        validation_images, transformation_idx_i)
                                categorical_trf_ids = tf.keras.utils.to_categorical(
                                    transformations_inds,
                                    num_classes=self.transformer.n_transforms)
                                self.eval_step(x_transformed,
                                               categorical_trf_ids)
                        template = 'Iter {}, Patience {}, Epoch {}, Loss: {}, Acc: {}, Val loss: {}, Val acc: {}, Time: {}'
                        print(
                            template.format(
                                iteration_i,
                                patience - self.best_model_so_far[
                                    general_keys.NOT_IMPROVED_COUNTER],
                                epoch + 1,
                                self.train_loss.result(),
                                self.train_accuracy.result() * 100,
                                self.eval_loss.result(),
                                self.eval_accuracy.result() * 100,
                                delta_timer(
                                    time.time() - self.training_star_time)
                            )
                        )
                        self.check_best_model_save(iteration_i)
                        self.eval_loss.reset_states()
                        self.eval_accuracy.reset_states()
                        self.train_loss.reset_states()
                        self.train_accuracy.reset_states()
        self.load_weights(
            self.best_model_weights_path)
        print('Total Training Time: {}'.format(
            delta_timer(time.time() - self.training_star_time)))
        self.print_manager.close()

    def predict(self, x, batch_size=1024):
        eval_ds = tf.data.Dataset.from_tensor_slices((x)).batch(batch_size)
        predictions = []
        for images in eval_ds:
            predictions.append(self.call_wrapper_to_predict(images))
        return np.concatenate(predictions, axis=0)

    def eval_tf(self, x, batch_size=1024, verbose=True):
        self.print_manager.verbose_printing(verbose)
        self.verbose = verbose
        self.eval_loss.reset_states()
        self.eval_accuracy.reset_states()
        dataset = tf.data.Dataset.from_tensor_slices(
            (x)).batch(batch_size)
        start_time = time.time()
        for transformation_idx_i in range(
            self.transformer.n_transforms):
            for images in dataset:
                transformations_inds = \
                    [transformation_idx_i] * images.shape[0]
                x_transformed = \
                    self.transformer.apply_specific_transform(images,
                                                              transformation_idx_i)
                categorical_trf_ids = tf.keras.utils.to_categorical(
                    transformations_inds,
                    num_classes=self.transformer.n_transforms)
                self.eval_step(x_transformed, categorical_trf_ids)
        template = 'Loss: {}, Acc: {}, Time: {}'
        print(template.format(
            self.eval_loss.result(),
            self.eval_accuracy.result() * 100,
            delta_timer(time.time() - start_time)
        ))
        results_dict = {general_keys.LOSS: self.eval_loss.result(),
                        general_keys.ACCURACY: self.eval_accuracy.result()}
        self.eval_loss.reset_states()
        self.eval_accuracy.reset_states()
        self.print_manager.close()
        return results_dict


if __name__ == '__main__':
    from modules.geometric_transform.streaming_transformers.transformer_ranking import \
        RankingTransformer
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from modules.utils import set_soft_gpu_memory_growth
    set_soft_gpu_memory_growth()

    EPOCHS = 1000
    ITERATIONS_TO_VALIDATE = None  # 1000 # None
    PATIENCE = 5  # 0

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
    transformer.set_transformations_to_perform(transformer.transformation_tuples*100)
    print(transformer.n_transforms)

    mdl = StreamingTransformationsDeepHits(transformer)
    # mdl.save_initial_weights(x_train, 'aux_weights')
    mdl.fit(
        x_train, epochs=EPOCHS, x_validation=x_val, batch_size=128,
        patience=PATIENCE, iterations_to_validate=ITERATIONS_TO_VALIDATE)
    mdl.eval_tf(x_train)
    mdl.eval_tf(x_val)
    print('\nResults with random Initial Weights')
    mdl.load_weights('aux_weights/init.ckpt')
    mdl.eval_tf(x_train)
    mdl.eval_tf(x_val)
    mdl.fit(
        x_train, epochs=EPOCHS, x_validation=x_val, batch_size=128,
        patience=PATIENCE, iterations_to_validate=ITERATIONS_TO_VALIDATE)
    mdl.eval_tf(x_train)
    mdl.eval_tf(x_val)

    del mdl
    mdl = StreamingTransformationsDeepHits(transformer)
    mdl.load_weights('aux_weights/best_weights.ckpt')
    print('\nResults with model loaded')
    mdl.eval_tf(x_train)
    mdl.eval_tf(x_val)
