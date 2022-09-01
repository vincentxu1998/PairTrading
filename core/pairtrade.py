from datetime import datetime
import pandas as pd
import sqlite3 as db
import itertools as it
import os.path as path
from os import makedirs
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from core.dataloader import DataLoader
from core.detrendPrice import detrendPrice
from core.utils import image_output_dir
import numpy as np
from core.WhiteRealityCheckFor1 import bootstrap


class PairTrade:
    '''
    Class for building the PairTrade
    '''

    def __init__(self,config):
        self.config = config
        self.plot_inline = config['plot_inline']
        self.intercept = config['intercept']
        self.window_spread_reg = self.config['window_spread_reg']
        self.entryZscore = self.config['entryZscore']
        self.exitZscore = self.config['exitZscore']
        self.allow_short = self.config['allow_short']
        self.allow_long = self.config['allow_long']
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']
        self.transactionFeeIn = self.config['transactionFeeIn']
        self.transactionFeeOut = self.config['transactionFeeOut']
        self.metrics = None
        self.wrc_result = None


    def build_engine(self, df1):
        logging.info("[MODEL]: Building engine...")

        #find spread and hedge ratio
        if self.config['regress_on_x']:
            self._apply_residual_model(df1)
            df1['spread'] = df1.y - df1.rolling_hedge_ratio * df1.x
        else:
            df1["rolling_hedge_ratio"] = 1
            df1['spread'] = np.log(df1.y) - np.log(df1.x)

        #find spread mean
        if self.config['spread_reg_on_time']:
            self._apply_spread_reg_on_time(df1, self.window_spread_reg)
        else:
            df1['meanSpread'] = df1['spread'].rolling(window=self.window_spread_reg).mean()

        stdSpread = df1.spread.rolling(window=self.window_spread_reg).std()
        df1['zScore'] = (df1.spread - df1.meanSpread) / stdSpread

        self._create_trading_strategy(df1)
        self._compute_returns(df1)
        self.metrics = self._compute_metrics(df1, count_trading=True)

        return df1

    def _create_trading_strategy(self, df1):
        df1['long entry'] = ((df1.zScore < - self.entryZscore))
        df1['long exit'] = ((df1.zScore > - self.exitZscore))

        #first consecutive long and exit index
        df1['long entry head'] = self._head_pos(df1['long entry'])
        df1['long exit head'] = self._head_pos(df1['long exit'])

        df1['num units long'] = np.nan
        df1.loc[df1['long entry'], 'num units long'] = 1
        df1.loc[df1['long exit'], 'num units long'] = 0
        df1.iat[0, df1.columns.get_loc("num units long")] = 0
        df1['num units long'] = df1['num units long'].fillna(method='pad')

        df1['long commision rate'] = 0
        df1.loc[df1['long entry head'], 'long commision rate'] = self.transactionFeeIn
        df1.loc[df1['long exit head'], 'long commision rate'] = self.transactionFeeOut


        df1['short entry'] = ((df1.zScore > self.entryZscore))
        df1['short exit'] = ((df1.zScore < self.exitZscore))

        df1['short entry head'] = self._head_pos(df1['short entry'])
        df1['short exit head'] = self._head_pos(df1['short exit'])

        df1['num units short'] = np.nan
        df1.loc[df1['short entry'], 'num units short'] = -1
        df1.loc[df1['short exit'], 'num units short'] = 0
        df1.iat[0, df1.columns.get_loc("num units short")] = 0
        df1['num units short'] = df1['num units short'].fillna(method='pad')

        df1['short commision rate'] = 0
        df1.loc[df1['short entry head'], 'short commision rate'] = self.transactionFeeIn
        df1.loc[df1['short exit head'], 'short commision rate'] = self.transactionFeeOut



        ###############################################################################################################################
        df1['numUnits'] = self.allow_long * df1['num units long'] + self.allow_short * df1['num units short']

        df1["positions_x"] = -1 * df1["rolling_hedge_ratio"] * df1["x"] * df1["numUnits"]
        df1["positions_y"] = df1["y"] * df1["numUnits"]

    def _head_pos(self, serie):
        return serie & ~(serie.shift(1).fillna(False))

    def _compute_returns(self, df1):
        '''
        Assumes that the signal is for that day i.e. if a signal of
        1 exists on the 12th of January, I should buy before that day begins
        '''


        df1['x_mkt_returns'] = df1['x'].pct_change(1)
        df1['y_mkt_returns'] = df1['y'].pct_change(1)

        df1['x_mkt_returns_com'] = df1['x'].pct_change(1) * (1- df1['short commision rate'].shift(1)) * (1- df1['long commision rate'].shift(1))
        df1['y_mkt_returns_com'] = df1['y'].pct_change(1) * (1- df1['short commision rate'].shift(1)) * (1- df1['long commision rate'].shift(1))
        df1["portfolio_cost"] = np.abs(df1["positions_x"]) + np.abs(df1["positions_y"])
        df1['commision_in'] = self.transactionFeeIn * df1['short entry head'] * df1['portfolio_cost'] \
                           + self.transactionFeeIn * df1['long entry head'] * df1['portfolio_cost']
        df1['commision_in'] = df1['commision_in'].shift(1)
        df1['commision_out'] = self.transactionFeeOut * df1['short exit head'] * df1['portfolio_cost'].shift(1) \
                           + self.transactionFeeOut * df1['long exit head'] * df1['portfolio_cost'].shift(1)
        df1['commision'] = df1['commision_in'] + df1['commision_out']


        df1['pnl_x'] = df1['x_mkt_returns'] * df1['positions_x'].shift(1)
        df1['pnl_y'] = df1['y_mkt_returns'] * df1['positions_y'].shift(1)

        df1['pnl_x_com'] = df1['x_mkt_returns_com'] * df1['positions_x'].shift(1)
        df1['pnl_y_com'] = df1['y_mkt_returns_com'] * df1['positions_y'].shift(1)

        df1["pnl"] = df1["pnl_x"] + df1["pnl_y"]


        df1["pnl_com"] = df1['pnl_x_com'] + df1['pnl_y_com'] - df1['commision']


        df1["port_rets"] = df1["pnl"] / df1["portfolio_cost"].shift(1)
        df1["port_rets-commision"] = df1["pnl_com"] / df1["portfolio_cost"].shift(1)

        df1["port_rets"].fillna(0, inplace=True)
        df1["port_rets-commision"].fillna(0, inplace=True)

        df1['system_equity'] = np.cumprod(1 + df1['port_rets'])
        df1['system_equity-commision'] = np.cumprod(1 + df1['port_rets-commision'])
        # df1.to_csv('check.csv')

    def _compute_metrics(self, df1, count_trading):
        start_date = df1.iloc[0].name
        end_date = df1.iloc[-1].name
        days = (end_date - start_date).days

        if count_trading:
            day_per_year = 252
        else:
            day_per_year = 360

        TotaAnnReturn = (df1.system_equity.tail(n=1) - 1) / (days / day_per_year)
        TotaAnnReturn_commision = (df1['system_equity-commision'].tail(n=1) - 1) / (days / day_per_year)

        system_cagr = (df1.system_equity.tail(n=1)) ** (day_per_year / days) - 1
        system_cagr_commision = (df1['system_equity-commision'].tail(n=1)) ** (day_per_year / days) - 1

        system_sharpe = np.sqrt(day_per_year) * np.mean(df1.port_rets) / np.std(df1.port_rets)
        system_sharpe_commision = np.sqrt(day_per_year) * np.mean(df1["port_rets-commision"]) / np.std(df1["port_rets-commision"])

        system_metrics = dict(
            TotaAnnReturn=TotaAnnReturn,
            system_cagr= system_cagr,
            system_sharpe=system_sharpe)
        system_metrics_tradding = dict(
            TotaAnnReturn_trading=TotaAnnReturn_commision,
            system_cagr_trading=system_cagr_commision,
            system_sharpe_trading=system_sharpe_commision)

        # if system_sharpe > .5:
        logging.info("TotaAnnReturn = %.4f" % round((TotaAnnReturn * 100),4))
        logging.info("CAGR = %.4f" % round((system_cagr * 100), 4))
        logging.info("Sharpe Ratio = %.4f" % (round(system_sharpe, 2)))

        logging.info("TotaAnnReturn after commision = %.4f" % round((TotaAnnReturn_commision * 100),4))
        logging.info("CAGR after commision= %.4f" % round((system_cagr_commision * 100), 4))
        logging.info("Sharpe Ratio after commision= %.4f" % (round(system_sharpe_commision, 2)))

        return system_metrics_tradding

    def _apply_residual_model(self, df1, window_hr_reg= 58):
        a = np.array([np.nan] * len(df1))
        b = [np.nan] * len(df1)  # If betas required.
        y_ = df1["y"].values
        x_ = df1[['x']].assign(
            constant=self.intercept).values  # if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
        for n in range(window_hr_reg, len(df1)):
            y = y_[(n - window_hr_reg):n]
            X = x_[(n - window_hr_reg):n]
            # betas = Inverse(X'.X).X'.y
            betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
            y_hat = betas.dot(x_[n, :])
            a[n] = y_hat
            b[n] = betas.tolist()

        myList = []
        for e in range(len(b)):
            if e < window_hr_reg:
                myList.append(0)
            else:
                myList.append(b[e][0])

        df1["rolling_hedge_ratio"] = myList

        # return df1

    def WRC(self, df1):
        '''
        detrend y and x,
        Apply white reality check on x and y
        '''

        df_detrend = df1.assign(x = detrendPrice(df1.x).values,
                         y = detrendPrice(df1.y).values)
        if self.config['regress_on_x']:
            self._apply_residual_model(df_detrend)
        else:
            df_detrend["rolling_hedge_ratio"] = 1

        self._compute_returns(df_detrend)
        self.wrc_result = bootstrap(df_detrend.port_rets)

    def _apply_spread_reg_on_time(self, df1, window= 7):
        a = np.array([np.nan] * len(df1))
        b = [np.nan] * len(df1)  # If betas required.
        y_ = df1['spread'].values
        x_ = df1[['TIME']].assign(
            constant=self.intercept).values  # if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
        for n in range(window, len(df1)):
            y = y_[(n - window):n]
            X = x_[(n - window):n]
            # betas = Inverse(X'.X).X'.y
            betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
            y_hat = betas.dot(x_[n, :])
            a[n] = y_hat
            b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept

        df1['meanSpread'] = a
        # return df1

    def plot_pair_price(self, df1, y_column, x_column):
        image_dir_path = path.join(image_output_dir, 'temp', y_column + '-' + x_column)
        if not path.exists(image_dir_path):
            makedirs(image_dir_path)

        plt.figure()
        plt.plot(df1.y, label=y_column)
        plt.plot(df1.x, label=x_column)
        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.title('%s vs %s' % (y_column, x_column))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if self.plot_inline:
            plt.show()
        else:
            plt.savefig(path.join(image_dir_path, y_column + '-' + x_column + 'Price.png'))
        plt.close()

        sns.jointplot(x=df1.x, y=df1.y, color='b')
        plt.title('%s vs %s' % (y_column, x_column))
        if self.plot_inline:
            plt.show()
        else:
            plt.savefig(path.join(image_dir_path,  y_column + '-' + x_column + 'ScatterPrice.png'))
        plt.close()

        plt.figure()
        plt.plot(df1.spread[self.window_spread_reg:], marker='o')
        plt.plot(df1.meanSpread[self.window_spread_reg:], marker='o', c='r')
        plt.title('%s vs %s' % (y_column, x_column))
        plt.title('spread: %s vs %s' % (y_column, x_column))
        if self.plot_inline:
            plt.show()
        else:
            plt.savefig(path.join(image_dir_path, y_column + '-' + x_column + 'Spread.png'))
        plt.close()

        plt.figure()
        df1['zScore'][self.window_spread_reg:].plot(marker='o')
        plt.title('zscore: %s vs %s' % (y_column, x_column))
        if self.plot_inline:
            plt.show()
        else:
            plt.savefig(path.join(image_dir_path, y_column + '-' + x_column + 'zScore.png'))
        plt.close()

    def plot_equity_curve(self, df1, y_column, x_column):
        image_dir_path = path.join(image_output_dir, 'equity_curve')
        if not path.exists(image_dir_path):
            makedirs(image_dir_path)
        plt.figure()
        plt.plot(df1['system_equity'])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title('EquityCurve')

        if self.plot_inline:
            plt.show()
        else:
            plt.savefig(path.join(image_dir_path, y_column + '-' + x_column + 'EquityCurve.png'))
        plt.close()
        
        image_dir_path = path.join(image_output_dir, 'equity_curve-commision')
        if not path.exists(image_dir_path):
            makedirs(image_dir_path)
        plt.figure()
        plt.plot(df1['system_equity-commision'])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title('EquityCurve-commission')

        if self.plot_inline:
            plt.show()
        else:
            plt.savefig(path.join(image_dir_path, y_column + '-' + x_column + 'EquityCurve-commission.png'))
        plt.close()

    def get_metrics(self):
        return self.metrics

    def get_wrc_result(self):
        return self.wrc_result




