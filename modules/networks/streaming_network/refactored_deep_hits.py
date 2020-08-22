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
import datetime
import numpy as np
from parameters import loader_keys, general_keys
from modules import utils
from modules.print_manager import PrintManager


# TODO: manage weights saved in a better manner: save_path, non saving when not given, etc
class DeepHits(tf.keras.Model):

    def __init__(
        self, n_classes, drop_rate=0.5, final_activation='softmax',
        name='deep_hits_refactored', results_folder_name=None):
        super().__init__(name=name)
        self.results_folder_path, self.best_model_weights_path = \
            self._create_model_paths(results_folder_name)
        self._init_layers(n_classes, drop_rate, final_activation)
        self._init_builds()


    def _init_layers(self, n_classes, drop_rate, final_activation):
        self.zp = tf.keras.layers.ZeroPadding2D(padding=(3, 3))
        self.conv_1 = tf.keras.layers.Conv2D(
            32, (4, 4), strides=(1, 1), padding='valid', activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.mp_1 = tf.keras.layers.MaxPool2D()
        self.conv_3 = tf.keras.layers.Conv2D(
            64, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(
            64, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv_5 = tf.keras.layers.Conv2D(
            64, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.mp_2 = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu')
        self.do_1 = tf.keras.layers.Dropout(drop_rate)
        self.dense_2 = tf.keras.layers.Dense(64, activation='relu')
        self.do_2 = tf.keras.layers.Dropout(drop_rate)
        self.dense_3 = tf.keras.layers.Dense(n_classes)
        self.act_out = tf.keras.layers.Activation(final_activation)

    def _init_builds(self):
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.eval_loss = tf.keras.metrics.Mean(name='eval_loss')
        self.eval_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='eval_accuracy')
        self.train_summary_writer = tf.summary.create_file_writer(
            os.path.join(self.results_folder_path, 'tensorboard/train'))
        self.val_summary_writer = tf.summary.create_file_writer(
            os.path.join(self.results_folder_path, 'tensorboard/val'))

    def _reset_metrics(self):
        self.eval_loss.reset_states()
        self.eval_accuracy.reset_states()

    def call(self, input_tensor, training=False, remove_top=False):
        x = self.zp(input_tensor)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.mp_1(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.mp_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.do_1(x, training=training)
        x = self.dense_2(x)
        x = self.do_2(x, training=training)
        if remove_top:
            return x
        x = self.dense_3(x)
        x = self.act_out(x)
        return x

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.call(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        accuracy = tf.reduce_mean(tf.cast(
            tf.argmax(labels, axis=-1) == tf.argmax(predictions, axis=-1),
            tf.float32))
        return loss, accuracy

    @tf.function
    def eval_step(self, images, labels):
        predictions = self.call(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.eval_loss(t_loss)
        self.eval_accuracy(labels, predictions)

    # # TODO: do something to keep model with best weights
    # def _fit_without_validation(self, x, y, batch_size, epochs):
    #     self.evaluation_set_name = 'train'
    #     train_ds = tf.data.Dataset.from_tensor_slices(
    #         (x, y)).shuffle(10000).batch(batch_size, drop_remainder=True)
    #     for epoch in range(epochs):
    #         epoch_start_time = time.time()
    #         for it_i, (images, labels) in enumerate(train_ds):
    #             self.train_step(images, labels)
    #             self.eval_step(images, labels)
    #         template = 'Epoch {}, Loss: {}, Acc: {}, Time: {}'
    #         print(template.format(epoch,
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

    # TODO: implement some kind of train_loggin
    def fit(self, x, y, epochs, validation_data=None, batch_size=128,
        iterations_to_validate=None, patience=None, verbose=True):
        print_manager = PrintManager().verbose_printing(verbose)
        print('\nTraining Initiated')
        self.training_star_time = time.time()
        self.best_model_so_far = {
            general_keys.ITERATION: 0,
            general_keys.LOSS: 1e100,
            general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH: 0,
        }
        self.n_iterations_in_epoch = (len(y) // batch_size)
        if validation_data is None:
            return self._fit_without_validation(x, y, batch_size, epochs)
        assert patience is not None
        self.evaluation_set_name = 'validation'
        # check if validate at end of epoch
        if iterations_to_validate is None:
            # -1 comes from fact that iterations start at 0
            iterations_to_validate = self.n_iterations_in_epoch - 1
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x, y)).shuffle(10000).batch(batch_size, drop_remainder=True)
        validation_ds = tf.data.Dataset.from_tensor_slices(
            (validation_data[0], validation_data[1])).batch(1024)
        for epoch in range(epochs):
            for it_i, (images, labels) in enumerate(train_ds):
                step_loss, step_accuracy = self.train_step(images, labels)
                it_i = it_i + (epoch * self.n_iterations_in_epoch)
                if it_i % iterations_to_validate == 0:
                    for validation_images, validation_labels in validation_ds:
                        self.eval_step(validation_images, validation_labels)
                    template = 'Iter {}, Patience {}, Epoch {}, Loss: {}, Acc: {}, Val loss: {}, Val acc: {}, Time: {}'
                    print(
                        template.format(
                            it_i,
                            patience-self.best_model_so_far[
                                general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH],
                            epoch,
                            step_loss,
                            step_accuracy * 100,
                            self.eval_loss.result(),
                            self.eval_accuracy.result() * 100,
                            delta_timer(
                                time.time() - self.training_star_time)
                        )
                    )
                    self.check_best_model_save(it_i)
                    self._reset_metrics()
                    if self.check_early_stopping(patience):
                        print_manager.close()
                        return
        self.load_weights(
            self.best_model_weights_path)
        self._print_training_end()
        print_manager.close()

    def _print_training_end(self):
        print("\nBest model @ it %d.\nValidation loss %.6f" % (
            self.best_model_so_far[general_keys.ITERATION],
            self.best_model_so_far[general_keys.LOSS]))
        print('\nTotal training time: {}\n'.format(
            delta_timer(time.time() - self.training_star_time)))

    def check_early_stopping(self, patience):
        if self.best_model_so_far[
            general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] > patience:
            self.load_weights(
                self.best_model_weights_path)
            self._print_training_end()
            self._reset_metrics()
            return True
        return False

    def _create_model_paths(self, results_folder_name):
        if results_folder_name is None:
            results_folder_name = self.name
            results_folder_path = results_folder_name
        else:
            date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            results_folder_path = os.path.join(
                PROJECT_PATH, 'results', results_folder_name,
                '%s_%s' % (self.name, date))
            utils.check_path(results_folder_path)
        best_model_weights_path = os.path.join(
            results_folder_path, 'checkpoints', 'best_weights.ckpt')
        return results_folder_path, best_model_weights_path

    def _set_model_results_paths(self, result_folder_path):
        self.results_folder_path = os.path.join(
            result_folder_path)
        self.best_model_weights_path = os.path.join(
            self.results_folder_path, 'checkpoints', 'best_weights.ckpt')

    def check_best_model_save(self, iteration):
        self.best_model_so_far[
            general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] += 1
        if self.eval_loss.result() < self.best_model_so_far[general_keys.LOSS]:
            self.best_model_so_far[general_keys.LOSS] = self.eval_loss.result()
            self.best_model_so_far[
                general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] = 0
            self.best_model_so_far[general_keys.ITERATION] = iteration
            self.save_weights(self.best_model_weights_path)
            print("\nNew best %s model: %s %.4f @ it %d\n" % (
                self.evaluation_set_name, general_keys.LOSS,
                self.best_model_so_far[general_keys.LOSS],
                self.best_model_so_far[general_keys.ITERATION]), flush=True)

    @tf.function
    def call_wrapper_to_predict(self, x):
        return self.call(x)

    def predict(self, x, batch_size=1024):
        eval_ds = tf.data.Dataset.from_tensor_slices((x)).batch(batch_size)
        predictions = []
        for images in eval_ds:
            predictions.append(self.call_wrapper_to_predict(images))
        return np.concatenate(predictions, axis=0)

    def evaluate(self, x, y, batch_size=1024, verbose=True):
        print_manager = PrintManager().verbose_printing(verbose)
        self.verbose = verbose
        self.eval_loss.reset_states()
        self.eval_accuracy.reset_states()
        dataset = tf.data.Dataset.from_tensor_slices(
            (x, y)).batch(batch_size)
        start_time = time.time()
        for images, labels in dataset:
            self.eval_step(images, labels)
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
        print_manager.close()
        return results_dict

    def load_weights(self, filepath, by_name=False):
        super().load_weights(filepath, by_name).expect_partial()

    def save_initial_weights(self, x, folder_path):
        self.call(x[0][None, ...])
        self.save_weights(os.path.join(folder_path, 'init.ckpt'))


if __name__ == '__main__':
    from modules.geometric_transform.transformer_for_ranking import RankingTransformer
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from tensorflow.keras.utils import to_categorical
    EPOCHS = 100
    ITERATIONS_TO_VALIDATE = None #1000 # None
    PATIENCE = 5 # 0

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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

    # transformer = transformations_tf.Transformer()
    # transformer = transformations_tf.TransTransformer()
    transformer = RankingTransformer()
    transformer.set_transformations_to_perform(
        transformer.transformation_tuples * 40)
    print(transformer.n_transforms)

    x_train_transformed, transformations_inds = transformer.apply_all_transforms(
        x_train)
    x_val_transformed, transformations_inds_val = transformer.apply_all_transforms(
        x_val)
    mdl = DeepHits(n_classes=transformer.n_transforms)
    mdl.save_initial_weights(x_train, 'aux_weights')
    mdl.fit(
        x_train_transformed, to_categorical(transformations_inds),
        epochs=EPOCHS,
        validation_data=(
        x_val_transformed, to_categorical(transformations_inds_val)),
        batch_size=128, patience=PATIENCE,
        iterations_to_validate=ITERATIONS_TO_VALIDATE)
    mdl.evaluate(x_train_transformed, to_categorical(transformations_inds),
                 verbose=1)
    mdl.evaluate(x_val_transformed, to_categorical(transformations_inds_val),
                 verbose=1)
    print('\nResults with random Initial Weights')
    mdl.load_weights('aux_weights/init.ckpt')
    mdl.evaluate(x_train_transformed, to_categorical(transformations_inds),
                 verbose=1)
    mdl.evaluate(x_val_transformed, to_categorical(transformations_inds_val),
                 verbose=1)
    mdl.fit(
        x_train_transformed, to_categorical(transformations_inds),
        epochs=EPOCHS,
        validation_data=(
        x_val_transformed, to_categorical(transformations_inds_val)),
        batch_size=128, patience=PATIENCE,
        iterations_to_validate=ITERATIONS_TO_VALIDATE)
    mdl.evaluate(x_train_transformed, to_categorical(transformations_inds),
                 verbose=1)
    mdl.evaluate(x_val_transformed, to_categorical(transformations_inds_val),
                 verbose=1)

    del mdl
    mdl = DeepHits(n_classes=transformer.n_transforms)
    mdl.load_weights('checkpoints/best_weights.ckpt')
    print('\nResults with model loaded')
    mdl.evaluate(x_train_transformed, to_categorical(transformations_inds),
                 verbose=1)
    mdl.evaluate(x_val_transformed, to_categorical(transformations_inds_val),
                 verbose=1)