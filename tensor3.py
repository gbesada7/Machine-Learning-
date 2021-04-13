import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cancer_classification.csv')
sns.countplot(x = 'benign_0__mal_1', data= df)

df.corr()['benign_0__mal_1'].sort_values().plot(kind = 'bar')

X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#X_train.shape()

model = Sequential()

model.add(Dense(30, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))

#binary classification

model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

#model.fit(x = X_train, y = y_train, epochs = 600, validation_data = (X_test, y_test))

losses = pd.DataFrame(model.history.history)
losses.plot()