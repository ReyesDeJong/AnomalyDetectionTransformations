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
from modules.networks.streaming_network.streaming_transformations_deep_hits \
    import StreamingTransformationsDeepHits
from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
from modules.networks.train_step_tf2.wide_residual_network import ResnetBlock
from parameters import constants
from modules.print_manager import PrintManager

WEIGHT_DECAY = 0.5 * 0.0005

# TODO: manage weights saved in a better manner: save_path, non saving when not
#  given, etc
class StreamingTransformationsWideResnet(StreamingTransformationsDeepHits):

    def __init__(self, input_channels, transformer: AbstractTransformer,
        depth=10, widen_factor=4, drop_rate=0.0, weight_decay=WEIGHT_DECAY,
        final_activation='softmax', name='WRN_Streaming_Trfs',
        results_folder_name=None):
        tf.keras.Model.__init__(self, name=name)
        self.print_manager = PrintManager()
        self.transformer = transformer
        self.input_channels = input_channels
        self.weight_decay = weight_decay
        self._init_layers(
            self.transformer.n_transforms, depth, widen_factor, drop_rate,
            final_activation)
        self._init_builds()
        self.results_folder_path, self.best_model_weights_path = \
            self._create_model_paths(results_folder_name)

    def _init_layers(self, n_classes, depth, widen_factor, dropout_rate,
        final_activation):
        self.regularizer = tf.keras.regularizers.l2(self.weight_decay)
        self.conv_initializer = tf.keras.initializers.VarianceScaling(
            scale=2.0, mode=constants.FAN_IN,
            distribution=constants.UNTRUNCATED_NORMAL)
        self.dense_initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0, mode=constants.FAN_IN,
            distribution=constants.UNIFORM)
        n_channels = [16, 16 * widen_factor, 32 * widen_factor,
                      64 * widen_factor]
        assert ((depth - 4) % 6 == 0), 'depth should be 6n+4'
        n_residual_blocks = (depth - 4) // 6
        self.conv_1 = self.conv2d(n_channels[0], 3, 1)
        self.group_1 = self.conv_group(
            self.input_channels, n_channels[1], n_residual_blocks, 1,
            dropout_rate)
        self.group_2 = self.conv_group(
            self.input_channels, n_channels[2], n_residual_blocks, 2,
            dropout_rate)
        self.group_3 = self.conv_group(
            self.input_channels, n_channels[3], n_residual_blocks, 2,
            dropout_rate)
        self.bn_1 = self.batch_norm()
        self.act_1 = tf.keras.layers.Activation('relu')
        self.gap_1 = tf.keras.layers.GlobalAveragePooling2D()
        self.fc_1 = self.dense(n_classes)
        self.act_out = tf.keras.layers.Activation(final_activation)

    def call(self, input_tensor, training=False):
        x = self.conv_1(input_tensor)
        x = self.group_1(x, training=training)
        x = self.group_2(x, training=training)
        x = self.group_3(x, training=training)
        x = self.bn_1(x, training=training)
        x = self.act_1(x)
        x = self.gap_1(x)
        x = self.fc_1(x)
        x = self.act_out(x)
        return x

    def conv_group(self, in_channels, out_channels, n_res_blocks, strides,
        dropout_rate=0.0):
        blocks = []
        blocks.append(
            ResnetBlock(in_channels, out_channels, strides, dropout_rate))
        for _ in range(1, n_res_blocks):
            blocks.append(
                ResnetBlock(out_channels, out_channels, 1, dropout_rate))
        return tf.keras.Sequential(blocks)

    # TODO: move elsewhere to avoid code replication
    def conv2d(self, filters, kernel_size, strides):
        return tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides,
            padding=constants.SAME, use_bias=False,
            kernel_initializer=self.conv_initializer,
            kernel_regularizer=self.regularizer)

    # TODO: move elsewhere to avoid code replication
    def batch_norm(self):
        return tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.9, epsilon=1e-5,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer)

    # TODO: move elsewhere to avoid code replication
    def dense(self, units):
        return tf.keras.layers.Dense(
            units, kernel_initializer=self.dense_initializer,
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer)


if __name__ == '__main__':
    from modules.geometric_transform.\
        streaming_transformers.transformer_ranking import RankingTransformer
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

    mdl = StreamingTransformationsWideResnet(x_train.shape[:-1], transformer)
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
    results_folder_path = mdl.results_folder_path
    print(os.path.abspath(results_folder_path))

    del mdl
    mdl = StreamingTransformationsWideResnet(x_train.shape[:-1], transformer)
    mdl.load_weights(os.path.join(results_folder_path, 'checkpoints',
                                  'best_weights.ckpt'))
    print('\nResults with model loaded')
    # mdl.evaluate(x_train, batch_size=1000)
    # mdl.evaluate(x_train, batch_size=1000)
    mdl.evaluate(x_train)
    mdl.evaluate(x_train)
    mdl.evaluate(x_val)
    mdl.evaluate(x_val)
    # mdl.evaluate(x_val, batch_size=256)
