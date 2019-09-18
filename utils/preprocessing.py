import pandas as pd
from keras.utils import to_categorical
from tqdm import tqdm


def answer_to_one_hot(answer, vocabulary):
    """
    Create a one-hot vector. indexes 0 to 4 correspond to the result of the question, next indexes correspond to the
    type of the question.

    Parameters
    ----------
    answer : Pandas DataFrame
        a single answer (one line in the Pandas DataFrame corresponding to the dataset)
    vocabulary : array-like
        vocabulary (cf utils.app_utils.create_vocabulary())
    Returns
    -------
    array-like
        one-hot vector
    """
    X = [0 for _ in range(len(vocabulary))]
    X[vocabulary.index(answer.loc["question"])] = 1
    X[vocabulary.index(answer.loc["result"])] = 1
    return X


def create_X_Y(dataframe, vocabulary):
    """
    Create X input matrix for the lstm network and Y label vector.
    Parameters
    ----------
    dataframe : Pandas DataFrame
        DataFrame containing all the training examples (cf induce-data csv)
    vocabulary : array-like
        vocabulary (cf utils.app_utils.create_vocabulary())
    Returns
    -------
    X : array-like
        input matrix
    Y : array-like
        label vector
    """
    X = []
    Y = []
    users = list(dict.fromkeys(dataframe.loc[:, "user"]))
    for u in tqdm(users, desc='creating input matrix'):
        crt_usr_df = dataframe[dataframe.user == u]
        x = []
        row = None
        for i in range(len(crt_usr_df)):
            row = crt_usr_df.iloc[i, :]
            x.append(answer_to_one_hot(row, vocabulary))
        X.append(x)
        Y.append(row.loc["ageGroup"])
    return X, to_categorical(Y)
