"""
Final version of WRN classifier in tf2 format

taken from modules/networks/streaming_network/straming_transformation_wrn.py

WARNING! most models got the input_shape need wrong, they don't need it!,
conv_groups wrong on other methods; wrn and streaming_wrn,
although is wrong it does nothing in this cases
"""

import os
import sys
import tensorflow as tf

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from parameters import loader_keys
from modules import utils
from modules.networks.deep_hitsv2 import DeepHitsv2
from parameters import constants
from modules.networks.train_step_tf2.wide_residual_network import ResnetBlock

WEIGHT_DECAY = 0.5 * 0.0005


class WideResnetv2(DeepHitsv2):

    def __init__(self, n_classes, depth=10, widen_factor=4, drop_rate=0.0,
        weight_decay=WEIGHT_DECAY, final_activation='softmax', name='WRNv2',
        results_folder_name=None):
        tf.keras.Model.__init__(self, name=name)
        self.weight_decay = weight_decay
        self.results_folder_path, self.best_model_weights_path = \
            self._create_model_paths(results_folder_name)
        self._init_layers(n_classes, depth, widen_factor, drop_rate,
                          final_activation)
        self._init_builds()

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
            n_channels[0], n_channels[1], n_residual_blocks, 1,
            dropout_rate)
        self.group_2 = self.conv_group(
            n_channels[1], n_channels[2], n_residual_blocks, 2,
            dropout_rate)
        self.group_3 = self.conv_group(
            n_channels[2], n_channels[3], n_residual_blocks, 2,
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
    # from modules.geometric_transform.transformer_for_ranking import \
    #     RankingTransformer
    from modules.data_loaders.hits_outlier_loaderv2 import HiTSOutlierLoaderv2
    from tensorflow.keras.utils import to_categorical
    from modules.geometric_transform import transformations_tf

    # TRAIN params
    EPOCHS = 100
    ITERATIONS_TO_VALIDATE = None  # 1000 # None
    PATIENCE = 0

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
    mdl = WideResnetv2(n_classes=transformer.n_transforms,
                       results_folder_name='wrnv2_tinkering')
    mdl.save_initial_weights(x_train, os.path.join(PROJECT_PATH, 'results',
                                                   'wrnv2_tinkering'))
    # print n layers in residual block, there is no error
    # print(len(mdl.layers[1].layers[0].layers)))
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
        os.path.join(PROJECT_PATH, 'results', 'wrnv2_tinkering',
                     'init.ckpt'))
    mdl.evaluate(x_train_transformed, to_categorical(transformations_inds),
                 verbose=True)
    mdl.evaluate(x_val_transformed, to_categorical(transformations_inds_val),
                 verbose=True)
    mdl.fit(
        x_train_transformed, to_categorical(transformations_inds),
        epochs=2)
    mdl.evaluate(x_train_transformed, to_categorical(transformations_inds),
                 verbose=True)
    mdl.evaluate(x_val_transformed, to_categorical(transformations_inds_val),
                 verbose=True)

    del mdl
    mdl = WideResnetv2(n_classes=transformer.n_transforms)
    mdl.load_weights(
        os.path.join(PROJECT_PATH, 'results', 'wrnv2_tinkering',
                     'init.ckpt'))
    print('\nResults with model loaded')
    mdl.evaluate(x_train_transformed, to_categorical(transformations_inds),
                 verbose=True)
    mdl.evaluate(x_val_transformed, to_categorical(transformations_inds_val),
                 verbose=True)
