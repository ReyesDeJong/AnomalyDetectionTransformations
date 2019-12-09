from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
# import tensorflow as tf
# from tensorflow.train_step_tf2.layers import Conv2D, Dense, Dropout, Flatten, Input, \
#   MaxPool2D, ZeroPadding2D, Activation
# from tensorflow.train_step_tf2 import Model


def rotate(img):
  orig = img
  rot1 = tf.image.rot90(orig, k=1, name="rot_90")
  rot2 = tf.image.rot90(orig, k=2, name="rot_180")
  rot3 = tf.image.rot90(orig, k=3, name="rot_270")

  final = tf.concat([orig, rot1, rot2, rot3], axis=0)

  return final


def create_simple_network(input_shape, num_classes, dropout_rate=0.0,
    final_activation='softmax'):
  inp = Input(shape=input_shape)
  out = ZeroPadding2D(padding=(3, 3))(inp)
  out = Conv2D(32, (4, 4), strides=(1, 1), padding='valid', activation='relu')(
    out)
  out = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(
    out)
  out = MaxPool2D()(out)
  out = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(
    out)
  out = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(
    out)
  out = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(
    out)
  out = MaxPool2D()(out)
  out = Flatten()(out)
  out = Dense(64, activation='relu')(out)
  out = Dropout(dropout_rate)(out)
  out = Dense(64, activation='relu')(out)
  out = Dropout(dropout_rate)(out)
  out = Dense(num_classes)(out)
  out = Activation(final_activation)(out)

  return Model(inp, out)


# def create_deep_hits(input_shape, num_classes, dropout_rate=0.0,
#     final_activation='softmax'):
#   inp = Input(shape=input_shape)
#   out = ZeroPadding2D(padding=(3, 3))(inp)
#   out = rotate(out)
#   out = Conv2D(32, (4, 4), strides=(1, 1), padding='valid', activation='relu')(
#     out)
#   out = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(
#     out)
#   out = MaxPool2D()(out)
#   out = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(
#     out)
#   out = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(
#     out)
#   out = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(
#     out)
#   out = MaxPool2D()(out)
#   out = Flatten()(out)
#   out = Dense(64, activation='relu')(out)
#   out = Dropout(dropout_rate)(out)
#   out = Dense(64, activation='relu')(out)
#   out = Dropout(dropout_rate)(out)
#   out = Dense(num_classes)(out)
#   orig, rot1, rot2, rot3 = tf.split(out, 4, axis=0)
#   out = (orig + rot1 + rot2 + rot3) / 4
#   out = Activation(final_activation)(out)
#
#   return Model(inputs=inp, outputs=out)
