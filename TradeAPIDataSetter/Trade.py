import math
import yfinance as yf
import datetime as dt
import numpy as np
import pandas as pd
import requests
from TradeAPIDataSetter.secrets import Secrets
import matplotlib.pyplot as plt
import mysql.connector

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

plt.style.use('fivethirtyeight')


class Trade:
    def __init__(self, symbol):
        sec = Secrets()  # API key stored in secrets.py
        self.token = sec.api_token
        self.user = sec.username
        self.passwd = sec.password
        self.symbol = symbol
    #   self.data = Trade.connect_to_api(self)
        self.data = Trade.connect_to_yahoo(self)
        self.data = pd.DataFrame.from_dict(self.data)
        self.data = self.data.replace({np.nan: 0})

    # Returns full JSON from API giving Stock data
    def connect_to_api(self):
        url = f'http://api.marketstack.com/v1/eod?access_key={self.token}&symbols={self.symbol}&date_from=1998-07-14&date_to=2021-07-14&limit=1000'
        r = requests.get(url)
        data = r.json()
        return data['data']

    def connect_to_yahoo(self):
        tick = yf.Ticker(f'{self.symbol}')
        history = tick.history(period="max")
        return history

    # Needs to be adjusted for newest API changes, df.index doesn't work
    # df.date should work but needs some testing/improvement
    def display_time_series(self):
        df = self.data
        print(df)
        print(df['Open'])
        fig, ax = plt.subplots(5)
        ax[0].plot(df.date, df['Open'])
        ax[0].legend(['Open'])
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        ax[1].plot(df.date, df['High'])
        ax[1].legend(['High'])
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[2].plot(df.date, df['Low'])
        ax[2].legend(['Low'])
        ax[2].set_xticklabels([])
        ax[2].set_yticklabels([])
        ax[3].plot(df.date, df['Close'])
        ax[3].legend(['Close'])
        ax[3].set_xticklabels([])
        ax[3].set_yticklabels([])
        ax[4].plot(df.date, df['Volume'])
        ax[4].legend(['Volume'])
        ax[4].set_xticklabels([])
        ax[4].set_yticklabels([])
        # fig.suptitle(f'{self.symbol} Stock {df.index[0]} to {df.index[len(df) - 1]}')
        plt.show()

    def add_to_mysql(self):
        db = mysql.connector.connect(host='localhost',
                                     user=self.user,
                                     passwd=self.passwd)
        mycursor = db.cursor()
        mycursor.execute('USE python_stock')
        for i in range(0, len(self.data) - 1):
            dat = '"' + f'{self.data.index[i]}' + '"'  # The API format is weird and this makes it normal
            ope = '"' + f'{self.data["Open"][i]}' + '"'
            high = '"' + f'{self.data["High"][i]}' + '"'
            low = '"' + f'{self.data["Low"][i]}' + '"'
            close = '"' + f'{self.data["Close"][i]}' + '"'
            volume = '"' + f'{self.data["Volume"][i]}' + '"'
            dividends = '"' + f'{self.data["Dividends"][i]}' + '"'
            splits = '"' + f'{self.data["Stock Splits"][i]}' + '"'
            sql = 'INSERT INTO ' + self.symbol + '(date, open, high, low, close, ' \
                                                 'volume, dividends,  ' \
                                                 'stock_splits)' + \
                  ' VALUES (' + dat + ', ' + ope + ', ' + high + \
                  ', ' + low + ', ' + close + ', ' + volume + ', ' + dividends + ', ' + splits + ')'
            mycursor.execute(sql)
        # mycursor.execute(f'UPDATE {self.symbol} SET adj_high=NULL WHERE adj_high=0')
        db.commit()

    def create_mysql_table(self):
        db = mysql.connector.connect(host='localhost',
                                     user=self.user,
                                     passwd=self.passwd)
        mycursor = db.cursor()
        mycursor.execute('USE python_stock')
        mycursor.execute(f'CREATE TABLE {self.symbol} '
                         f'(id INT AUTO_INCREMENT PRIMARY KEY, '
                         f'symbol VARCHAR(5) DEFAULT "{self.symbol}", '
                         f'date TIMESTAMP DEFAULT NOW() UNIQUE, '
                         f'open DECIMAL(20,4), '
                         f'high DECIMAL(20,4), '
                         f'low DECIMAL(20,4), '
                         f'close DECIMAL(20,4), '
                         f'volume DECIMAL(30,4), '
                         f'dividends DECIMAL(20,4), '
                         f'stock_splits DECIMAL(20,4))')
        db.commit()

    def predict_close(self):
        df = self.data
        df = df.tail(7000) # number of days to train and test
        close = df['Close'].values
        # Normalize data to [0,1] to fit into neural network
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

        x_train = []
        y_train = []

        prediction_days = 1400 # should be 20% of len(df)

        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x-prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        # convert x and y train to numpy array
        x_train, y_train = np.array(x_train), np.array(y_train)
        # not entirely sure why this reshape is needed tbh. will debug and find out later
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Building model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Dense(units=1)) # Prediction of closing price
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=25, batch_size=32) # sees data 25 times and 32 units at once

        ''' TEST MODEL ACCURACY'''
        train_size = math.ceil(len(df['Close']) * 0.8)
        train_set = df['Close'].head(train_size).values
        test_set = df['Close'].tail(len(df['Close'])-train_size).values
        total_set = np.concatenate((train_set, test_set), axis=0)
        model_inputs = total_set[len(df) - len(test_set) - prediction_days:]
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)

        # Make predictions on test data
        x_test = []
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - prediction_days:x, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted_prices = model.predict(x_test)
        print(len(predicted_prices[:][:][0]))
        predicted_prices = scaler.inverse_transform(predicted_prices[:][:][0])

        # Plot the prediction results
        plt.plot(test_set, color='black')
        plt.plot(predicted_prices, color='green')
        # plt.legend()
        plt.show()

    def drop_table(self):
        db = mysql.connector.connect(host='localhost',
                                     user=self.user,
                                     passwd=self.passwd)
        mycursor = db.cursor()
        mycursor.execute('USE python_stock')
        mycursor.execute(f'DROP TABLE {self.symbol}')
        db.commit()


# Script
symbols = ['AAPL', 'MSFT', 'AMZN', 'FB',
           'GOOD', 'TSLA', 'NVDA', 'GOOGL',
           'PYPL', 'ADBE', 'CMCSA', 'NFLX',
           'INTC', 'CSCO', 'PEP', 'AVGO',
           'TMUS', 'COST', 'TXN', 'QCOM']

for symbol in symbols:
    trade = Trade(symbol)
    trade.create_mysql_table()
    trade.add_to_mysql()
    # trade.drop_table()

symbol = 'AAPL'
trade = Trade(symbol)
trade.predict_close()