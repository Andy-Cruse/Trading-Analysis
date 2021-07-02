import numpy as np
import pandas as pd
import requests
from TradeAPIDataSetter.secrets import Secrets
import matplotlib.pyplot as plt
import mysql.connector

# IN NEED OF SQL TABLE DUPLICATE REMOVER
class Trade:
    def __init__(self, symbol, interval):
        sec = Secrets()  # API key stored in secrets.py
        self.token = sec.api_token # Need to get your own API key. The one I used, AlphaVantage, was free
        self.user = sec.username 
        self.passwd = sec.password
        self.symbol = symbol
        self.interval = interval
        self.data = Trade.connect_to_api(self) # returns data as dictionary
        self.time_series = self.data[f'Time Series ({self.interval}min)']
        self.meta_data = self.data['Meta Data']
        self.data = pd.DataFrame.from_dict(self.data)
        self.data = self.data.transpose().sort_index()
        self.time_series = pd.DataFrame.from_dict(self.time_series)
        self.time_series = self.time_series.transpose().sort_index()
        # self.meta_data = pd.DataFrame.from_dict(self.meta_data) # Had issues running this line. The data is not needed for anything, but still...
        # self.meta_data = self.meta_data.transpose().sort_index()

    # Returns full JSON from API call
    def connect_to_api(self):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={self.symbol}&interval={self.interval}min&apikey={self.token}'
        r = requests.get(url)
        data = r.json()
        return data

    # Displays the time series for open, high, low, close, and volume
    def disp_time_series(self):
        df = self.time_series
        print(df)
        print(df['1. open'])
        fig, ax = plt.subplots(5)
        # There is likely a better way to do this
        ax[0].plot(df.index, df['1. open'])
        ax[0].legend(['Open'])
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        ax[1].plot(df.index, df['2. high'])
        ax[1].legend(['High'])
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[2].plot(df.index, df['3. low'])
        ax[2].legend(['Low'])
        ax[2].set_xticklabels([])
        ax[2].set_yticklabels([])
        ax[3].plot(df.index, df['4. close'])
        ax[3].legend(['Close'])
        ax[3].set_xticklabels([])
        ax[3].set_yticklabels([])
        ax[4].plot(df.index, df['5. volume'])
        ax[4].legend(['Volume'])
        ax[4].set_xticklabels([])
        ax[4].set_yticklabels([])
        fig.suptitle(f'{self.symbol} Stock {df.index[0]} to {df.index[len(df)-1]}')
        plt.show()

    # Creates a MySQL table
    # While commenting, I notice that it would be a good idea to set the database name to a variable. Will update
    def create_mysql_table(self):
        db = mysql.connector.connect(host='localhost',
                                     user=self.user,
                                     passwd=self.passwd)
        mycursor = db.cursor()
        mycursor.execute('USE python_stock')
        mycursor.execute(f'CREATE TABLE {self.symbol}_{self.interval}min '
                         f'(date VARCHAR(100), '
                         f'open VARCHAR(10), '
                         f'high VARCHAR(10), '
                         f'low VARCHAR(10), '
                         f'close VARCHAR(10), '
                         f'volume VARCHAR(10))')
        db.commit()

    
    # Adds data for each minute interval specified earlier to a MySQL table
    def add_to_mysql(self):
        db = mysql.connector.connect(host='localhost',
                                     user=self.user,
                                     passwd=self.passwd)
        mycursor = db.cursor()
        mycursor.execute('USE python_stock')
        for i in range(0, len(self.time_series)-1):
            index = '"'+f'{self.time_series.index[i]}'+'"'
            ope = '"'+f'{self.time_series["1. open"][i]}'+'"'
            high = '"'+f'{self.time_series["2. high"][i]}'+'"'
            low = '"'+f'{self.time_series["3. low"][i]}'+'"'
            close = '"'+f'{self.time_series["4. close"][i]}'+'"'
            volume = '"'+f'{self.time_series["5. volume"][i]}'+'"'
            sql = 'INSERT INTO '+self.symbol+'_'+self.interval+'min VALUES ('+index+', '+ope+', '+high+', '+low+', '+close+', '+volume+')'
            mycursor.execute(sql)
        db.commit()
        
        
    # Old method I used previously to make excel sheets. Excel is expensive.
    # Not needed, but I don't wanna toss it
    # def write_to_excel(self, dataframe):
    #     writer = pd.ExcelWriter('recommended trades.xlsx', engine='xlsxwriter')
    #     dataframe.to_excel(writer, 'Recommended Trades', index=False)
    #     background_color = '#0a0a23'
    #     font_color = '#ffffff'
    #     string_format = writer.book.add_format(
    #         {
    #             'font_color': font_color,
    #             'bg_color': background_color,
    #             'border': 1
    #         }
    #     )
    #     dollar_format = writer.book.add_format(
    #         {
    #             'num_format': '$0.00',
    #             'font_color': font_color,
    #             'bg_color': background_color,
    #             'border': 1
    #         }
    #     )
    #     integer_format = writer.book.add_format(
    #         {
    #             'num_format': '0',
    #             'font_color': font_color,
    #             'bg_color': background_color,
    #             'border': 1
    #         }
    #     )
    #     writer.sheets['Recommended Trades'].set_column('A:A', 18, string_format)
    #     writer.sheets['Recommended Trades'].set_column('B:B', 18, string_format)
    #     writer.sheets['Recommended Trades'].set_column('C:C', 18, string_format)
    #     writer.sheets['Recommended Trades'].set_column('D:D', 18, string_format)
    #     writer.save()



# Script
symbol = 'IBM'  # example symbol
interval = '5'
trade = Trade(symbol, interval)
trade.create_mysql_table()
trade.add_to_mysql()
trade.disp_time_series()

