# import packages
import pandas as pd
import numpy as np

# setting figure size
from fastai.tabular.core import add_datepart
from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10

# for normalizing data
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

scaler = MinMaxScaler(feature_range=(0, 1))

# read the file
df = pd.read_csv('NSE-TATAGLOBAL11.csv')
df['Data'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Data']
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Data', 'Close'])
print(data['Data'])
for i in range(0, len(data)):
    new_data['Data'][i] = data['Data'][i]
    new_data['Data'][i] = data['Close'][i]

# set index
new_data.index = new_data.Data
new_data.drop('Data', axis=1, inplace=True)

# create train and test sets
dataset = new_data.values
train = dataset[0:987, :]  # first 80%
valid = dataset[987:, :]  # last 20%

# convert dataset to x_train, y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fir the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# predicting 246 values, using past 60 from the train data
num = len(new_data)-len(valid) - 60
print(new_data.shape)
print(new_data[num:])
inputs = new_data[num:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
print(rms)
