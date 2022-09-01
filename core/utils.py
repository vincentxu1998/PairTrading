import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
import sys
from datetime import datetime,timezone
import os
import pytz

output_dir = 'output'
tz = pytz.timezone('America/Toronto')


def create_output_directory():
    now = datetime.now(tz=tz).strftime("output%y-%m-%d:%H:%M:%S")
    image_output_dir =os.path.join(output_dir, now)
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
    return image_output_dir

global image_output_dir
image_output_dir = create_output_directory()




def concatenate_strat_to_test(test_df, trading_signal, seq_len):
    '''
    Concatenates the trading signal to the test_df
    '''
    new_df = test_df.copy()
    
    # Start and stop length. Start at the seq_len or lookback window - 1
    # This is because if the lookback window is set to 27, we are looking
    # at the last 26 and then predicting for the 27th
    new_signal = np.hstack(([np.nan], trading_signal))
    start = seq_len-2
    stop = start + len(new_signal)
    
    # Add the signal to the dataframe
    new_df = new_df.iloc[start:stop, :]
    new_df['signal'] = new_signal.reshape(-1,1)
    
    return new_df

def compute_returns(df, price_col):
    '''
    Assumes that the signal is for that day i.e. if a signal of 
    1 exists on the 12th of January, I should buy before that day begins
    '''
    new_df = df.copy()
    
    new_df['mkt_returns'] = new_df[price_col].pct_change(1)
    new_df['system_returns'] = new_df['mkt_returns']*new_df['signal']
    
    new_df['system_equity'] = np.cumprod(1+new_df.system_returns) - 1
    new_df['mkt_equity'] = np.cumprod(1+new_df.mkt_returns) - 1
    
    return new_df

def plot_returns(df):
    ax = df[['system_equity','mkt_equity']].plot(figsize=(100, 50), linewidth=4.0)
    ax.tick_params(labelsize=88)
    plt.savefig(os.path.join(image_output_dir, 'returns'), dpi=100)


def compute_metrics(df):
    new_df = df.copy()
    new_df['system_equity']=np.cumprod(1+new_df.system_returns) -1
    system_cagr=(1+new_df.system_equity.tail(n=1))**(252/new_df.shape[0])-1
    new_df.system_returns= np.log(new_df.system_returns+1)
    system_sharpe = np.sqrt(252)*np.mean(new_df.system_returns)/np.std(new_df.system_returns)

    new_df['mkt_equity']=np.cumprod(1+new_df.mkt_returns) -1
    mkt_cagr=(1+new_df.mkt_equity.tail(n=1))**(252/new_df.shape[0])-1
    new_df.mkt_returns= np.log(new_df.mkt_returns+1)
    mkt_sharpe = np.sqrt(252)*np.mean(new_df.mkt_returns)/np.std(new_df.mkt_returns)
    
    system_metrics = (system_cagr, system_sharpe)
    market_metrics = (mkt_cagr, mkt_sharpe)
    
    return system_metrics, market_metrics