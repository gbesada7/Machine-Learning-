import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('kc_house_data.csv')

#df.isnull().sum() #check for missing data, get the count 
#print(df.describe().transpose()) # statistical info from dataframe
plt.figure(figsize = (10,6))
#sns.distplot(df['price'])#distribution of data, most houses falling between 0 and 1.5million dollars, some outliers
#print(df.corr()['price'].sort_values()) #high corr of sq ft and price
#sns.scatterplot(x = 'price', y = 'sqft_living', data = df)

X = df.drop('price', axis = 1).values
X = df.drop('date', axis = 1)
y = df['price'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(19, activation = 'relu'))
model.add(Dense(19, activation = 'relu'))
model.add(Dense(19, activation = 'relu'))
model.add(Dense(19, activation = 'relu'))

model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')

model.fit(x = X_train, y = y_train, validation_data= (X_test, y_test),
          batch_size = 128, epochs = 400)#large data set feed it in in batches


losses = pd.DataFrame(model.history.history)
#loss decreases after iterations
losses.plot()

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

predictions = model.predict(X_test)
mean_squared_error(y_test, predictions)
mean_absolute_error(y_test, predictions)
np.sqrt(mean_squared_error)
df['price'].describe()

explained_variance_score(y_test, predictions)
plt.figure(figsize = (12, 6))
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')

single_house = df.drop('price', axis = 1).iloc[0]
scaler.transform(single_house.values.reshape(-1, 19))
model.predict(single_house)

