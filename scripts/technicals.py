

import pandas as pd
import numpy as np


def cal_ema(series:pd.Series):
    period = len(series)
    return series.ewm(span=period,min_periods=period, adjust=True).mean().iloc[-1]

def cal_rsi(series:pd.Series):
    period=len(series)
    delta = series.diff()
    
    ## positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # EMAs of ups and downs
    _gain = up.ewm(span=period, adjust=True).mean()
    _loss = down.abs().ewm(span=period, adjust=True).mean()

    RS = _gain / _loss
    RSI = 100 - (100 / (1 + RS))

    return RSI.iloc[-1]



def cal_ssma(series:pd.Series):
    """
    Smoothed simple moving average.
    """
    period = len(series)
    return series.ewm(ignore_na=False, alpha=1.0 / period, min_periods=0, adjust=True).mean().iloc[-1]



def cal_dema(series:pd.Series):
    """
    Double Exponential Moving Average 
    """
    period = len(series)
    e_m_a = series.ewm(span=period,min_periods=period, adjust=True).mean()
    d_e_m_a = (2 * e_m_a - e_m_a).ewm(span=period, adjust=True).mean()


    return d_e_m_a.iloc[-1]


def cal_tema(series:pd.Series):
    """
    Triple exponential moving average 
    """
    period = len(series)
    e_m_a = series.ewm(span=period,min_periods=period, adjust=True).mean()
    triple_ema = 3 * e_m_a
    ema_ema_ema = (
        e_m_a
        .ewm(ignore_na=False, span=period, adjust=True)
        .mean()
        .ewm(ignore_na=False, span=period, adjust=True)
        .mean()
    )

    TEMA = (
        triple_ema
        - 3
        * e_m_a
        .ewm(span=period, adjust=True)
        .mean()
        + ema_ema_ema
    )

    return TEMA.iloc[-1]


def cal_trix(series:pd.Series):
    """
    The Triple Exponential Moving Average Oscillator (TRIX) 
    """

    period = int(len(series)/2)
    if period<1:
        period=len(series)
    EMA1 = series.ewm(span=period,min_periods=period, adjust=True).mean()
    EMA2 = EMA1.ewm(span=period, adjust=True).mean()
    EMA3 = EMA2.ewm(span=period, adjust=True).mean()
    TRIX = (EMA3 - EMA3.diff()) / EMA3.diff()

    return TRIX.iloc[-1]




def cal_er(series:pd.Series):
    period=int(len(series)/2)
    if period<1:
        period=len(series)
    change = series.diff(period).abs()
    volatility = series.diff().abs().rolling(window=period).sum()

    return (change / volatility).iloc[-1]




def cal_smma(series:pd.Series):

    period = int(len(series))
    return series.ewm(alpha=1 / period, adjust=True).mean().iloc[-1]



def cal_roc(series:pd.Series) -> pd.Series:
    period=int(len(series)/60)
    if period<1:
        period=len(series)
    return ((series.diff(period) / series.shift(period)) * 100).iloc[-1]



def cal_stochrsi(series:pd.Series):
       
    period=len(series)
    delta = series.diff()
    rsi_period=int(period/2)
    if rsi_period<1:
        rsi_period=len(series)
        
    stoch_period=int(period/4)
    if stoch_period<1:
        stoch_period=len(series)
    ## positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # EMAs of ups and downs
    _gain = up.ewm(span=period, adjust=True).mean()
    _loss = down.abs().ewm(span=period, adjust=True).mean()

    RS = _gain / _loss
    rsi = 100 - (100 / (1 + RS))
    
    return (
        ((rsi - rsi.min()) / (rsi.max() - rsi.min()))
        .rolling(window=stoch_period)
        .mean()).iloc[-1]



#---full data frame
def cal_wap(df_raw):
    return ((df_raw['bid_price1'] * df_raw['ask_size1'] +
                    df_raw['ask_price1'] * df_raw['bid_size1']) / (
                                           df_raw['bid_size1']+ df_raw['ask_size1'])).values
                                           

def cal_spread1(df_raw):
    return (df_raw['ask_price1']/df_raw['bid_price1']-1).values

def cal_spread2(df_raw):
    return (df_raw['ask_price2']/df_raw['bid_price2']-1).values                                           
#per time_id
def cal_log_return(prices):
    return np.log(prices).diff()
    
#per time_id   
def cal_real_vol(log_return):
    return np.sqrt(np.sum(log_return**2))