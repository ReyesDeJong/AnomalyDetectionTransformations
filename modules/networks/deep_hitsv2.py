"""
Final version of DeepHiTS classifier in tf2 format

taken from modules/networks/streaming_network/refactored_deep_hits.py

"""

import os
import sys

import tensorflow as tf

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.utils import delta_timer
import time
import datetime
import numpy as np
from parameters import loader_keys, general_keys
from modules import utils
from modules.print_manager import PrintManager


class DeepHitsv2(tf.keras.Model):

    def __init__(self, n_classes, drop_rate=0.5, name='deep_hitsv2',
        results_folder_name=None):
        super().__init__(name=name)
        self.results_folder_path, self.best_model_weights_path = \
            self._create_model_paths(results_folder_name)
        self._init_layers(n_classes, drop_rate)
        self._init_builds()

    def _init_layers(self, n_classes, drop_rate):
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
        self.act_out = tf.keras.layers.Activation('softmax')

    def _init_builds(self):
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.eval_loss = tf.keras.metrics.Mean(name='eval_loss')
        self.eval_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='eval_accuracy')

    def _init_tensorboard_summaries(self):
        self.train_summary_writer = tf.summary.create_file_writer(
            os.path.join(self.results_folder_path, 'tensorboard/train'))
        self.val_summary_writer = tf.summary.create_file_writer(
            os.path.join(self.results_folder_path, 'tensorboard/val'))

    def _reset_metrics(self):
        self.eval_loss.reset_states()
        self.eval_accuracy.reset_states()

    def call(self, input_tensor, training=False):
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

    #TODO: include print at iteration, logging in this case isof whole train set
    def _fit_without_validation(self, x, y, batch_size, epochs, print_manager,
        file):
        print('\nTraining Initiated')
        print(x.shape)
        self.training_star_time = time.time()
        # datasets preparation
        self.n_iterations_in_epoch = (len(y) // batch_size)
        x, y = self._shuffle_dataset(x, y)
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x, y)).shuffle(10000).batch(batch_size, drop_remainder=True)
        # training loop
        for epoch in range(epochs):
            for it_i, (images, labels) in enumerate(train_ds):
                self.train_step(images, labels)
            it_i = it_i + (epoch * self.n_iterations_in_epoch)
            evaluation_dict = self.evaluate(x, y, verbose=False)
            print('Time usage: %s' %
                    delta_timer(time.time() - self.training_star_time))
            self._print_at_train(
                evaluation_dict[general_keys.ACCURACY],
                evaluation_dict[general_keys.LOSS], it_i, epoch+1, 'dataset')
        self.save_weights(self.best_model_weights_path)
        print('\nTotal training time: {}\n'.format(
            delta_timer(time.time() - self.training_star_time)))
        print_manager.close()
        file.close()

    def _check_if_validation_at_epoch_end(self, n_iterations_in_epoch,
        iterations_to_validate):
        if iterations_to_validate:
            return iterations_to_validate
        # -1 comes from fact that iterations start at 0
        return n_iterations_in_epoch - 1

    def _shuffle_dataset(self, x, y):
        dataset_indexes = np.arange(len(y))
        np.random.shuffle(dataset_indexes)
        x = x[dataset_indexes]
        y = y[dataset_indexes]
        return x, y

    def _get_fit_print_manager_and_file(self, verbose, log_file):
        print_manager = PrintManager().verbose_printing(verbose)
        file = open(os.path.join(self.results_folder_path, log_file), 'w')
        print_manager.file_printing(file)
        return print_manager, file

    def fit(self, x, y, epochs, validation_data=None, batch_size=128,
        iterations_to_validate=None, patience=None, verbose=True,
        log_file='train.log', iterations_to_print_train=None):
        # init loggers
        self._reset_metrics()
        self._init_tensorboard_summaries()
        print_manager, file = self._get_fit_print_manager_and_file(verbose, log_file)
        if validation_data is None:
            return self._fit_without_validation(x, y, batch_size, epochs,
                                                print_manager, file)
        print('\nTraining Initiated')
        print(x.shape)
        # training variables preparation
        self.training_star_time = time.time()
        self.best_model_so_far = {
            general_keys.ITERATION: 0,
            general_keys.LOSS: 1e100,
            general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH: 0,
        }
        self.n_iterations_in_epoch = (len(y) // batch_size)
        iterations_to_validate = self._check_if_validation_at_epoch_end(
            self.n_iterations_in_epoch, iterations_to_validate)
        if iterations_to_print_train is None:
            iterations_to_print_train = iterations_to_validate
        # datasets preparation
        x, y = self._shuffle_dataset(x, y)
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x, y)).shuffle(10000).batch(batch_size, drop_remainder=True)
        validation_ds = tf.data.Dataset.from_tensor_slices(
            (validation_data[0], validation_data[1])).batch(1024)
        # training loop
        for epoch in range(epochs):
            for it_i, (images, labels) in enumerate(train_ds):
                it_i = it_i + (epoch * self.n_iterations_in_epoch)
                # validation
                if it_i % iterations_to_validate == 0:
                    self._validate(
                        validation_ds, it_i, patience, epoch)
                    if self.check_early_stopping(patience, print_manager, file):
                        return
                # train step
                step_loss, step_accuracy = self.train_step(images, labels)
                if it_i % iterations_to_print_train == 0:
                    self._print_at_train(
                        step_accuracy, step_loss, it_i, epoch, 'step')
        self.load_weights(
            self.best_model_weights_path)
        self._print_training_end()
        print_manager.close()
        file.close()

    def _print_at_train(self, accuracy, loss, iteration, epoch,
        data_amount_name):
        template = '(Train %s) Epoch %i Iteration %i: loss %.6f, acc %.6f'
        print(template % (
            data_amount_name,
            epoch,
            iteration,
            loss,
            accuracy * 100
        ))
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=iteration)
            tf.summary.scalar('accuracy', accuracy,
                              step=iteration)

    def _validate(
        self, validation_ds: tf.data.Dataset, iteration, patience, epoch):
        for validation_images, validation_labels in validation_ds:
            self.eval_step(validation_images, validation_labels)
        best_model_finding_message = self.check_best_model_save(iteration, self.eval_loss.result())
        template = 'Time usage: %s\n(Validation) Epoch %i Iteration %i ' \
                   'Patience left %i: loss %.6f, acc %.6f %s'
        print(template % (
            delta_timer(time.time() - self.training_star_time),
            epoch,
            iteration,
            patience - self.best_model_so_far[
                general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] + 1, # +1 is necessary because patience 0 allows 1 check
            self.eval_loss.result(),
            self.eval_accuracy.result() * 100,
            best_model_finding_message
        ))
        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', self.eval_loss.result(), step=iteration)
            tf.summary.scalar('accuracy', self.eval_accuracy.result(),
                              step=iteration)
        self._reset_metrics()

    def _print_training_end(self):
        print("\nBest model @ it %d.\nValidation loss %.6f" % (
            self.best_model_so_far[general_keys.ITERATION],
            self.best_model_so_far[general_keys.LOSS]))
        print('\nTotal training time: {}\n'.format(
            delta_timer(time.time() - self.training_star_time)))

    def check_early_stopping(self, patience, print_manager, file):
        if self.best_model_so_far[
            general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] > patience:
            self.load_weights(
                self.best_model_weights_path)
            self._print_training_end()
            self._reset_metrics()
            print_manager.close()
            file.close()
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

    def check_best_model_save(self, iteration, validation_loss):
        self.best_model_so_far[
            general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] += 1
        output_message = ''
        if validation_loss < self.best_model_so_far[general_keys.LOSS]:
            self.best_model_so_far[general_keys.LOSS] = validation_loss
            self.best_model_so_far[
                general_keys.COUNT_MODEL_NOT_IMPROVED_AT_EPOCH] = 0
            self.best_model_so_far[general_keys.ITERATION] = iteration
            self.save_weights(self.best_model_weights_path)
            output_message = "\n\nNew best validation model: %s %.6f @ it %d\n" % (
                general_keys.LOSS, self.best_model_so_far[general_keys.LOSS],
                self.best_model_so_far[general_keys.ITERATION])
        return output_message

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
        self._reset_metrics()
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
        self._reset_metrics()
        print_manager.close()
        return results_dict

    def load_weights(self, filepath, by_name=False):
        super().load_weights(filepath, by_name).expect_partial()

    def save_initial_weights(self, x, folder_path):
        utils.check_path(folder_path)
        self.call(x[0][None, ...])
        self.save_weights(os.path.join(folder_path, 'init.ckpt'))


if __name__ == '__main__':
    # from modules.geometric_transform.transformer_for_ranking import \
    #     RankingTransformer
    from modules.data_loaders.hits_outlier_loaderv2 import HiTSOutlierLoaderv2
    from tensorflow.keras.utils import to_categorical
    from modules.geometric_transform import transformations_tf

    # TRAIN params
    EPOCHS = 100
    ITERATIONS_TO_VALIDATE = None  # 1000 # None
    PATIENCE = 3  # 0

    utils.set_soft_gpu_memory_growth()
    # data load
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '..', 'datasets', 'thesis_data', 'hits',
            'hits_small_4c_tuples.pkl'),
    }
    data_loader = HiTSOutlierLoaderv2(hits_params, 'small_hits')
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = data_loader.get_outlier_detection_datasets()

    # data transformation
    transformer = transformations_tf.TransTransformer()
    print('n transformations: ', transformer.n_transforms)
    x_train_transformed, transformations_inds = transformer.apply_all_transforms(
        x_train)
    x_val_transformed, transformations_inds_val = transformer.apply_all_transforms(
        x_val)

    # model training
    mdl = DeepHitsv2(n_classes=transformer.n_transforms,
                   results_folder_name='deep_hitsv2_tinkering')
    mdl.save_initial_weights(x_train, os.path.join(PROJECT_PATH, 'results',
                                                   'deep_hitsv2_tinkering'))
    mdl.fit(
        x_train_transformed, to_categorical(transformations_inds),
        epochs=EPOCHS,
        validation_data=(
            x_val_transformed, to_categorical(transformations_inds_val)),
        batch_size=128, patience=PATIENCE,
        iterations_to_validate=ITERATIONS_TO_VALIDATE)
    mdl.evaluate(x_train_transformed, to_categorical(transformations_inds),
                 verbose=True)
    mdl.evaluate(x_val_transformed, to_categorical(transformations_inds_val),
                 verbose=True)
    print('\nResults with random Initial Weights')
    mdl.load_weights(
        os.path.join(PROJECT_PATH, 'results', 'deep_hitsv2_tinkering',
                     'init.ckpt'))
    mdl.evaluate(x_train_transformed, to_categorical(transformations_inds),
                 verbose=True)
    mdl.evaluate(x_val_transformed, to_categorical(transformations_inds_val),
                 verbose=True)
    mdl.fit(
        x_train_transformed, to_categorical(transformations_inds),
        epochs=5)
    mdl.evaluate(x_train_transformed, to_categorical(transformations_inds),
                 verbose=True)
    mdl.evaluate(x_val_transformed, to_categorical(transformations_inds_val),
                 verbose=True)

    del mdl
    mdl = DeepHitsv2(n_classes=transformer.n_transforms)
    mdl.load_weights(
        os.path.join(PROJECT_PATH, 'results', 'deep_hitsv2_tinkering',
                     'init.ckpt'))
    print('\nResults with model loaded')
    mdl.evaluate(x_train_transformed, to_categorical(transformations_inds),
                 verbose=True)
    mdl.evaluate(x_val_transformed, to_categorical(transformations_inds_val),
                 verbose=True)
