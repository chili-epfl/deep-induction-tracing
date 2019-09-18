import pandas as pd


def answer_to_one_hot(answer, vocabulary):
    X = [0 for _ in range(len(vocabulary))]
    X[vocabulary.index(answer.loc["question"])] = 1
    X[vocabulary.index(answer.loc["result"])] = 1
    return X
