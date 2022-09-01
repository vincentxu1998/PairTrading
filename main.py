from datetime import datetime
import pandas as pd
import sqlite3 as db
import itertools as it
import logging
from core.utils import *
from core.pairtrade import PairTrade
from core.dataloader import DataLoader


logging.basicConfig(level=logging.INFO)

config = dict(entryZscore = 1,
              exitZscore = -0,
              window = 7,
              spread_reg_on_time = 1,
              regress_on_x = 0,
              intercept = 1,
              allow_short = 1,
              allow_long = 1,
              start_date = datetime.strptime('2012-01-01', '%Y-%m-%d'),
              end_date = datetime.now(),
              which_first = 'yfirst',
              plot_inline = False,
              window_spread_reg = 7,
              transactionFeeIn = 0.001,
              transactionFeeOut = 0.002
              )

database = 'input/sqlite/PythonData.db'
# sql = 'SELECT Ticker FROM etftable WHERE "Asset Class" = "Currency";'
# sql = 'SELECT Ticker FROM etftable WHERE "Niche" = "Energy";'
# sql = 'SELECT Ticker FROM etftable WHERE "Niche" = "Broad-based";'
sql = 'SELECT Ticker FROM etftable WHERE "Focus" = "Silver";'

# create a connection to the database specified above
cnx = db.connect(database)
cur = cnx.cursor()

tickers = pd.read_sql(sql, con=cnx)

symbList = sorted(list(set([tickers.iloc[i][0] for i in range(len(tickers))])))
# symbList = ['SPY', 'DIA']
# symbList = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'USDC-USD', 'BNB-USD', 'BUSD-USD', 'XRP-USD',
#             'ADA-USD', 'SOL-USD', 'DOGE-USD', 'HEX-USD', 'DOT-USD', 'SHIB-USD', 'DAI-USD', 'WTRX-USD',
#             'AVAX-USD', 'MATIC-USD', 'TRX-USD', 'STETH-USD', 'WBTC-USD', 'UNI1-USD', 'ETC-USD', 'LTC-USD',
#             'FTT-USD', 'BIT1-USD']
symbList = ['ETH-USD', 'STETH-USD']

r = {}
msg = ""
for pair in list(it.combinations(symbList, 2)):
    ret = (0,0,0)
    try:
        logging.info('---------------------------------------')
        logging.info('%s - %s' % pair)
        pairDL = DataLoader(pair, config)
        tradeEngine = PairTrade(config)

        df1 = tradeEngine.build_engine(pairDL.df1)
        tradeEngine.plot_pair_price(df1, pairDL.y_column, pairDL.x_column)

        #pair that generates sharpe ratio > 0.5 is filter out
        if tradeEngine.get_metrics()['system_sharpe_trading'] > 0.5:
            tradeEngine.WRC(df1)

            #pair that pass the WRC will be collected in the equity curve directory
            if tradeEngine.get_wrc_result()['flag']:
                tradeEngine.plot_equity_curve(df1, pairDL.y_column, pairDL.x_column)
                logging.info('White Reality Check  %s - %s: PASS' % (pair[0], pair[1]))
                logging.info('AVG return on detrend data: %.4f' % (tradeEngine.get_wrc_result()['average_return']))
            else:
                logging.info('White Reality Check %s - %s: FAIL' % (pair[0], pair[1]))

    except Exception as e:
        logging.error(e)
        raise



