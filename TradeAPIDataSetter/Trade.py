import numpy as np
import pandas as pd
import requests
import xlsxwriter
import math
from secrets import Secrets


class Trade:
    def __init__(self):
        sec = Secrets() #API key stored in secrets.py
        self.stocks = pd.read_csv('sp_500_stocks.csv') 
        self.token = sec.api_token

    ### Returns full JSON from API call
    def connect_to_api(self, symbol):
        base_url = f'https://cloud.iexapis.com/stable/stock/{symbol}/quote?token={self.token}'
        data = requests.get(base_url).json()
        return data

    def panda_frame(self):
        trade = Trade()
        my_columns = ['Ticker', 'Stock Price', 'Market Capitalization', 'Number of Shares to Buy']
        final_dataframe = pd.DataFrame(columns=my_columns)
        for stock in self.stocks['Ticker']:
            data = trade.connect_to_api(stock)
            final_dataframe = final_dataframe.append(
                pd.Series(
                    [
                        stock,
                        data['latestPrice'],
                        data['marketCap'],
                        trade.shares_to_buy(1000)
                    ],
                    index = my_columns),
                ignore_index = True
            )
        return final_dataframe

    def shares_to_buy(self, portfolio_size):
        return portfolio_size

    def write_to_excel(self, dataframe):
        writer = pd.ExcelWriter('recommended trades.xlsx', engine ='xlsxwriter')
        dataframe.to_excel(writer, 'Recommended Trades', index = False)
        background_color = '#0a0a23'
        font_color = '#ffffff'
        string_format = writer.book.add_format(
            {
                'font_color': font_color,
                'bg_color': background_color,
                'border': 1
            }
        )
        dollar_format = writer.book.add_format(
            {
                'num_format': '$0.00',
                'font_color': font_color,
                'bg_color': background_color,
                'border': 1
            }
        )
        integer_format = writer.book.add_format(
            {
                'num_format': '0',
                'font_color': font_color,
                'bg_color': background_color,
                'border': 1
            }
        )
        writer.sheets['Recommended Trades'].set_column('A:A', 18, string_format)
        writer.sheets['Recommended Trades'].set_column('B:B', 18, string_format)
        writer.sheets['Recommended Trades'].set_column('C:C', 18, string_format)
        writer.sheets['Recommended Trades'].set_column('D:D', 18, string_format)
        writer.save()


x = Trade()
x.write_to_excel(x.panda_frame())
