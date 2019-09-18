import pandas as pd
from keras import Sequential
from keras.layers import CuDNNLSTM, Dropout, Dense, Activation

from utils.app_utils import create_vocabulary
from utils.preprocessing import create_X_Y


def train_lstm():
    dataframe = pd.read_csv("induce-data-2019-08-08.csv")
    voc = create_vocabulary()
    X, Y = create_X_Y(dataframe, voc, ['8-10', '11-13'])
    print(X.shape)
    model = lstm(X, 2)
    model.fit(X, Y, epochs=1000, batch_size=10, verbose=1)


def lstm(X, n):
    """Build an LSTM RNN"""
    model = Sequential()
    model.add(CuDNNLSTM(
        41,
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=False
    ))
    model.add(Dense(n))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


if __name__ == "__main__":
    train_lstm()
