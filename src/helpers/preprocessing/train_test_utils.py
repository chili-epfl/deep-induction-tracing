from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

from src.helpers.preprocessing.sequences import split_sequence

vocab = [
    '_O',
    '_C',
    '_T',
    'OT',
    'CO',
    'CT', ]

labels = ['correct',
          'wrong',
          'type',
          'orientation',
          'color']

types = ['INTRO',
         'CORE',
         'FLEX',
         'TRIK',
         'DELY'
         ]

topics = ['cards',
          'animals',
          'geometry'

          ]

feat = ['type',
        'color',
        'orientation',
        'dual'
        ]

age = ['8-10', '11-13']


def prepare_sequences(data, type_=None, feature=None):
    """
    Prepare inputs for training.

    Parameters
    ----------
    data : Pandas.DataFrame
    type_ : str
                Default = None. Question type (CORE, INTRO,...)
    feature : str
        Default = None. Question topic (cards, animals, geometry).

    Returns
    -------
    X : array-like
        Training points (features)
    y : array-like
        Training points (labels)
    """
    if type_ is not None:
        data = data[data.type == type_]
    if feature is not None:
        data = data[data.topic == feature]

    data['question'] = data['question'].apply(lambda x: x[-2] + x[-1])

    y = list()
    X = list()

    for i in tqdm(range(2, 63)):
        X_seq, y_seq = split_sequence(data, i, vocab, labels, types, feat, topics, age)
        for x in X_seq:
            X.append(x)
        for _y in y_seq:
            y.append(_y)

    y = to_categorical(np.asarray(y))
    X = np.asarray(X)
    X = pad_sequences(X, value=-1, maxlen=63)

    return X, y

def prepare_and_test(data_test, model, type_=None, feature=None, reshape_cnn=False, reshape_conv=False):
    if type_ is not None:
        data_test = data_test[data_test.type == type_]
    if feature is not None:
        data_test = data_test[data_test.topic == feature]

    data_test['question'] = data_test['question'].apply(lambda x: x[-2] + x[-1])

    y_test = list()
    X_test = list()

    for i in tqdm(range(2, 63)):
        X_seq, y_seq = split_sequence(data_test, i, vocab, labels, types, feat, topics, age)
        for x in X_seq:
            X_test.append(x)
        for _y in y_seq:
            y_test.append(_y)

    y_test = to_categorical(np.asarray(y_test))
    X_test = np.asarray(X_test)
    X_test = pad_sequences(X_test, value=-1, maxlen=63)
    print(model.metrics_names)

    if reshape_cnn:
        n_steps, n_length = 7, 9
        n_features = 7
        X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))
    if reshape_conv:
        n_steps, n_length = 7, 9
        n_features = 7
        X_test = X_test.reshape((X_test.shape[0], n_steps, 1, n_length, n_features))
    print(model.evaluate(X_test, y_test))
