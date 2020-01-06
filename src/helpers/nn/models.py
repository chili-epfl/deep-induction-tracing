from keras import regularizers
from keras import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, Activation, Masking, BatchNormalization, CuDNNLSTM, TimeDistributed, \
    LeakyReLU, CuDNNGRU, SimpleRNN
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers import ConvLSTM2D

from src.helpers.metrics.scores import f1_m


def lstm(X, dropout_rate=0.15, n_labels=5, hidden_units=100, gpu=False, stacked=False, amsgrad_on=False):
    amsgrad = Adam(amsgrad=amsgrad_on)
    model = Sequential()
    if gpu:
        model.add(
            CuDNNLSTM(hidden_units, return_sequences=stacked, stateful=False, input_shape=(X.shape[1], X.shape[2]),
                      kernel_regularizer=regularizers.l1(0.01)))
    else:
        model.add(Masking(-1., input_shape=(X.shape[1], X.shape[2])))
        model.add(
            LSTM(hidden_units, return_sequences=stacked, stateful=False, input_shape=(X.shape[1], X.shape[2]),
                 kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(dropout_rate))

    if stacked:
        if gpu:
            model.add(CuDNNLSTM(hidden_units, return_sequences=False))
            model.add(Dropout(dropout_rate))
        else:
            model.add(LSTM(hidden_units, return_sequences=False))
            model.add(Dropout(dropout_rate))

    model.add(Dense(n_labels))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=amsgrad, metrics=['accuracy', f1_m])

    return model


def gru(X, dropout_rate=0.15, n_labels=5, hidden_units=100, gpu=False, amsgrad_on=False):
    amsgrad = Adam(amsgrad=amsgrad_on)
    model = Sequential()

    if gpu:
        model.add(
            CuDNNGRU(hidden_units, return_sequences=False, input_shape=(X.shape[1], X.shape[2]),
                     kernel_regularizer=regularizers.l1(0.01)))
    else:
        model.add(Masking(-1., input_shape=(X.shape[1], X.shape[2])))
        model.add(
            GRU(hidden_units, return_sequences=False, input_shape=(X.shape[1], X.shape[2]),
                kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(dropout_rate))

    model.add(Dense(n_labels))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=amsgrad, metrics=['accuracy', f1_m])

    return model


def cnn_lstm(X, gpu=False, dropout_rate=0.15, n_labels=5, hidden_units=100, amsgrad_on=False):
    amsgrad = Adam(amsgrad=amsgrad_on)
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
                        input_shape=(None, X.shape[-2], X.shape[-1])))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(dropout_rate)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    if gpu:
        model.add(CuDNNLSTM(hidden_units))
    else:
        model.add(LSTM(hidden_units))
    model.add(Dropout(dropout_rate))

    model.add(Dense(n_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=amsgrad, metrics=['accuracy', f1_m])

    return model


def conv_lstm(X, dropout_rate=0.15, n_labels=5, amsgrad_on=False):
    amsgrad = Adam(amsgrad=amsgrad_on)

    model = Sequential()
    model.add(
        ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu', input_shape=(X.shape[1], 1, X.shape[3], X.shape[4])))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(n_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=amsgrad, metrics=['accuracy', f1_m])

    return model
