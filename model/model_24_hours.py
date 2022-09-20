from statistics import mode
import tensorflow as tf
keras = tf.keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import TimeDistributed


def model_24_hours(
    input_shape_0: int,
    input_shape_1: int,
    output_shape_0: int,
    output_shape_1: int,
    drop_out: float,
    n_LSTM: int,
    n_Dense: int,
    ):
    model = Sequential()
    model.add(Bidirectional(LSTM(n_LSTM//2,return_sequences= True), input_shape=(input_shape_0, input_shape_1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(LSTM(n_LSTM, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(RepeatVector(output_shape_0))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(TimeDistributed(Dense(n_Dense)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(TimeDistributed(Dense(output_shape_1)))

    return model