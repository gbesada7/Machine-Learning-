import pandas as pd
import numpy as np

import seaborn as sns

df = pd.read_csv('fake_reg.csv')

#sns.pairplot(df)

from sklearn.model_selection import train_test_split

X = df[['feature1', 'feature2']].values

y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_test)#scaled between 0 and 1
X_test = scaler.transform(X_test)


X_train.min()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
#3 layers with 4 neurons each to predict price based on the 2 features of the independent variables
model.add(Dense(4, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
#model.add(Dense(4, activation = 'relu'))

model.add(Dense(1))#price prediccted

model.compile(optimizer = 'rmsprop', loss = 'mse')

model.fit(x = X_train, y = y_train, epochs = 250)#250 iterations(epoch)
#mean squared error loss function will decrease after each iteration (gradient descent)

loss_df = pd.DataFrame(model.history)

loss_df.plot()

