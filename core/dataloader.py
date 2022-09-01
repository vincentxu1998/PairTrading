import logging

import pandas as pd
import numpy as np
import os.path as path
from os import makedirs

from core.utils import image_output_dir

pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import yfinance as yf



class DataLoader:
    '''
    Dataloader for loading stock price
    '''

    def __init__(self, pair, config):
        if config['which_first'] == 'yfirst':
            self.y_column = pair[0]
            self.x_column = pair[1]
        else:
            self.y_column = pair[1]
            self.x_column = pair[0]

        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.df1 = self._load_data()

    def _load_data(self):
        # image_output_dir = 'output/output22-08-31:01:04:39'
        stock_price_directory = path.join(image_output_dir, 'stocks_csv')
        if not path.exists(stock_price_directory):
            makedirs(stock_price_directory)
        try:
            y = pdr.get_data_yahoo(self.y_column, start=self.start_date, end=self.end_date)
            y.to_csv(path.join(stock_price_directory, self.y_column + '.csv'), header = True, index=True, encoding='utf-8')
            x = pdr.get_data_yahoo(self.x_column, start=self.start_date, end=self.end_date)
            x.to_csv(path.join(stock_price_directory, self.x_column + '.csv'), header = True, index=True, encoding='utf-8')
        except Exception as e:
            msg = "yahoo problem"
            logging.error(msg)


        y = pd.read_csv(path.join(stock_price_directory, self.y_column + '.csv'), parse_dates=['Date'])
        y = y.sort_values(by='Date')
        y.set_index('Date', inplace = True)
        x = pd.read_csv(path.join(stock_price_directory,self.x_column + '.csv'), parse_dates=['Date'])
        x = x.sort_values(by='Date')
        x.set_index('Date', inplace = True)

        y.rename(
            columns={'Open': 'y_Open', 'High': 'y_High', 'Low': 'y_Low', 'Close': 'y_Close', 'Adj Close': 'y_Adj_Close',
                     'Volume': 'y_Volume'}, inplace=True)
        x.rename(
            columns={'Open': 'x_Open', 'High': 'x_High', 'Low': 'x_Low', 'Close': 'x_Close', 'Adj Close': 'x_Adj_Close',
                     'Volume': 'x_Volume'}, inplace=True)
        df1 = pd.merge(x, y, left_index=True, right_index=True, how='inner')

        # get rid of extra columns but keep the date index
        df1.drop(
            ['x_Open', 'x_High', 'x_Low', 'x_Close', 'x_Volume', 'y_Open', 'y_High', 'y_Low', 'y_Close', 'y_Volume'],
            axis=1, inplace=True)
        df1.rename(columns={'y_Adj_Close': 'y', 'x_Adj_Close': 'x'}, inplace=True)
        df1 = df1.assign(TIME=pd.Series(np.arange(df1.shape[0])).values)
        return df1

    def get_data(self):
        return self.df1






