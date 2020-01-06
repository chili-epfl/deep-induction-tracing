from keras.utils import plot_model
import matplotlib.pyplot as plt


def save_graph(model, name):
    """
    Wrapper for the keras.utils plot_model function.
    Parameters
    ----------
    model : Keras.Model
    name : str
        Path to be saved (filename).
    """
    return plot_model(model, to_file=name, show_shapes=True, show_layer_names=False)


def plot_loss(history):
    """
    Plot training and validation loss
    Parameters
    ----------
    history : Keras.History
        Fitted model history
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')


def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
