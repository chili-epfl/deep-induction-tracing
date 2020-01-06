import numpy as np
import pandas as pd



def seq_to_int(qts, vocab, labels, types, feat, topics, n_steps, age):
    """
    Convert sequences to an array of categorical integers.

    Parameters
    ----------
    qts : Pandas Series
        Set of questions from the dataset.
    vocab : list
    labels : list
    types : list
    feat : list
    topics list
    n_steps : int
        size of the sliding window
    age : list

    Returns
    -------
    list
        list of question vectors.
    """
    integ = list()
    for i, x in enumerate(qts):
        if i != n_steps:
            features = list()
            features.append(vocab.index(qts[i, 4]))
            features.append(qts[i, 2])
            features.append(types.index(qts[i, 9]))
            features.append(feat.index(qts[i, 10]))
            features.append(topics.index(qts[i, 6]))
            features.append(age.index(qts[i, 7]))
            features.append(labels.index(qts[i, 5]))
        else:
            features = list()
            features.append(vocab.index(qts[i, 4]))
            features.append(qts[i, 2])
            features.append(types.index(qts[i, 9]))
            features.append(feat.index(qts[i, 10]))
            features.append(topics.index(qts[i, 6]))
            features.append(age.index(qts[i, 7]))
            features.append(-1)
        integ.append(features)
    return integ


def split_sequence(data, n_steps, vocab, labels, types, feat, topics, age):
    """
    Apply a sliding window as described on the report, on the dataset.
    Parameters
    ----------
    data : Pandas.DataFrame
        Loaded dataset using pd.read_csv()
    n_steps : int
        size of the sliding window
    vocab : list
    labels : list
    types : list
    feat : list
    topics : list
    age : list

    Returns
    -------
    X : array-like
        Training points (features)
    y : array-like
        Training points (labels)
    """
    X, Y = list(), list()
    users = list(dict.fromkeys(data.loc[:, "user"]))
    for u in users:
        sequence = data[data.user == u]
        for i in range(len(sequence)):
            end_idx = i + n_steps
            if end_idx > len(sequence) - 1:
                break
            x = seq_to_int(sequence.values[i:end_idx + 1, :], vocab, labels, types, feat, topics, n_steps, age)
            y = labels.index(str(sequence.values[end_idx, 5]))
            X.append(x)
            Y.append(y)
    return np.array(X), np.array(Y)
