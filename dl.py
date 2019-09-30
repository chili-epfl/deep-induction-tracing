# %%
import pandas as pd
import numpy as np
from tqdm import tqdm

# %%
data = pd.read_csv('induce-data-2019-08-08.csv')
vocab = ['void',
         'C_E_F_T',
         'C_E_F_C',
         'C_E_F_O',
         'A_E_F_T',
         'A_E_F_O',
         'A_E_F_C',
         'G_E_F_C',
         'G_E_F_T',
         'G_E_F_O',
         'A_E_M_T',
         'A_E_M_O',
         'A_E_M_C',
         'G_E_M_O',
         'G_E_M_C',
         'G_E_M_T',
         'C_E_M_O',
         'C_E_M_C',
         'C_E_M_T',
         'C_H_F_CO',
         'C_H_F_CT',
         'C_H_F_OT',
         'G_H_F_OT',
         'G_H_F_CO',
         'G_H_F_CT',
         'A_H_F_CT',
         'A_H_F_OT',
         'A_H_F_CO',
         'C_H_M_CO',
         'C_H_M_CT',
         'C_H_M_OT',
         'A_H_M_CT',
         'A_H_M_OT',
         'A_H_M_CO',
         'G_H_M_OT',
         'G_H_M_CO',
         'G_H_M_CT', ]
labels = ['correct',
          'wrong',
          'type',
          'orientation',
          'color']


def qts_to_seq(qts, vocab, labels):
    x = np.zeros(63)
    y = np.zeros(5)
    for i in range(len(qts)):
        x[i] = vocab.index(qts.iloc[i, 4])
    y = labels.index(str(qts.iloc[len(qts) - 1, 5]))
    return x, y


print(qts_to_seq(data.iloc[0:5, :], vocab, labels))

# %%
X = []
Y = []
users = list(dict.fromkeys(data.loc[:, "user"]))
for u in tqdm(users, desc='creating input matrix'):

    crt_usr_df = data[data.user == u]
    for i in range(len(crt_usr_df)):
        for j in range(2, 63):
            x, y = qts_to_seq(crt_usr_df.iloc[0:j, :], vocab, labels)
            X.append(x)
            Y.append(y)
print(np.array(X).shape)
print(np.array(Y).shape)

# %%
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split

X = np.array(X)
Y = np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)
model = Sequential()
model.add(Embedding(len(vocab), 63, input_length=63))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
