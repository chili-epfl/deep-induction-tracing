import pandas as pd


def answer_to_one_hot(answer, vocabulary):
    """
    Create a one-hot vector. indexes 0 and 1 correspond to the result of the question, next indexes correspond to the
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
