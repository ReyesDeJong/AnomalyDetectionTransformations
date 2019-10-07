from keras.layers import *
from keras.models import Model


def create_simple_network(input_shape, num_classes, dropout_rate=0.0,
    final_activation='softmax'):
  inp = Input(shape=input_shape)
  out = ZeroPadding2D(padding=(3, 3))(inp)
  out = Conv2D(32, 4, 1, activation='relu')(out)
  out = Conv2D(32, 3, 1, activation='relu')(out)
  out = MaxPooling2D()(out)
  out = Conv2D(64, 3, 1, activation='relu')(out)
  out = Conv2D(64, 3, 1, activation='relu')(out)
  out = Conv2D(64, 3, 1, activation='relu')(out)
  out = MaxPooling2D()(out)
  out = Flatten()(out)
  out = Dense(64, activation='relu')(out)
  out = Dropout(dropout_rate)(out)
  out = Dense(64, activation='relu')(out)
  out = Dropout(dropout_rate)(out)
  out = Dense(num_classes)(out)
  out = Activation(final_activation)(out)

  return Model(inp, out)
