import pandas as pd

from src.helpers.plots.keras_plots import plot_loss, plot_acc
from src.helpers.preprocessing.train_test_utils import prepare_sequences
from src.helpers.nn.models import *

TRAIN_PATH = '../generated/train.csv'
VALIDATION_PATH = '../generated/validation.csv'
TYPE = 'CORE'
FEATURE = 'geometry'
GPU = False
BATCH_SIZE = 1000
EPOCHS = 500


def preprocessing():
    "Wrapper for the preprocessing routine"

    # Load datasets
    data_train = pd.read_csv(TRAIN_PATH)
    data_validation = pd.read_csv(VALIDATION_PATH)
    # Train/Validation split
    X_train, y_train = prepare_sequences(data_train, type_=TYPE, feature=FEATURE)

    X_validation, y_validation = prepare_sequences(data_validation, type_=TYPE, feature=FEATURE)

    return X_train, y_train, X_validation, y_validation


def train(name, X_train, y_train, X_validation, y_validation):
    "Wrapper for the training routine"
    if name == 'vanilla_lstm':
        model = lstm(X_train, dropout_rate=0.3, hidden_units=100, gpu=GPU)
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, y_validation),
                  shuffle=True)
        return model, history

    elif name == 'stacked_lstm':
        model = lstm(X_train, dropout_rate=0.15, hidden_units=100, gpu=GPU, stacked=True)
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, y_validation),
                  shuffle=True)
        return model, history
    elif name == 'gru':
        model = gru(X_train, dropout_rate=0.3, hidden_units=100, gpu=GPU)
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, y_validation),
                  shuffle=True)
        return model, history
    elif name == 'cnnlstm':
        n_steps, n_length = 7, 9
        n_features = 7
        X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))
        X_validation = X_validation.reshape((X_validation.shape[0], n_steps, n_length, n_features))

        model = cnn_lstm(X_train, dropout_rate=0.3, gpu=GPU)
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, y_validation),
                  shuffle=True)
        return model, history
    elif name == 'convlstm':
        n_steps, n_length = 7, 9
        n_features = 7
        X_train = X_train.reshape((X_train.shape[0], n_steps, 1, n_length, n_features))
        X_validation = X_validation.reshape((X_validation.shape[0], n_steps, 1, n_length, n_features))

        model = conv_lstm(X_train, dropout_rate=0.3)
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, y_validation),
                  shuffle=True)
        return model, history


if __name__ == '__main__':
    # Preprocess data
    X_train, y_train, X_validation, y_validation = preprocessing()
    # Train model
    model, history = train('stacked_lstm', X_train, y_train, X_validation, y_validation)
    # Save model
    model.save('../generated/model.h5')
    # Plots
    plot_loss(history)
    plot_acc(history)


