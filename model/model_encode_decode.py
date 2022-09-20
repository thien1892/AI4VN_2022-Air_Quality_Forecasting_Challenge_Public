import tensorflow as tf
keras = tf.keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import Activation


def model_encode_decode(
    n_steps_in,
    n_features,
    n_steps_out,
    n_features_out,
    n_LSTM,
    n_Dense,
    drop_out ):
    model = Sequential()
    model.add(LSTM(n_LSTM, input_shape=(n_steps_in, n_features), return_sequences= True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(LSTM(n_LSTM,return_sequences= True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(LSTM(n_LSTM))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(n_Dense))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(RepeatVector(n_steps_out))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(LSTM(n_LSTM, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(LSTM(n_LSTM, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(TimeDistributed(Dense(n_Dense)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(TimeDistributed(Dense(n_Dense)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    model.add(TimeDistributed(Dense(n_features_out)))
    return model