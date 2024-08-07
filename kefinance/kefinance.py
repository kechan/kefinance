from typing import Union

import os, sys, time, pickle, urllib.request, requests, json, re, tempfile, gc
from pathlib import *
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np

import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError

from io import StringIO

from .common.utils import load_from_pickle, save_to_pickle

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ALPHAVANTAGE_API_KEY=''

def stringify(in_date):
  return in_date.strftime('%Y-%m-%d')

today = datetime.today()
today_s = stringify(today)
today = datetime.strptime(today_s, '%Y-%m-%d').date()      # today should be stripped of current hrs, min, sec.

if today.weekday() == 5:
  weekend_offset = timedelta(days=-1)
elif today.weekday() == 6:
  weekend_offset = timedelta(days=-2)
else:
  weekend_offset = timedelta(days=0)

fifteen_year_ago = (today - timedelta(weeks=52*15))
twelve_year_ago = (today - timedelta(weeks=52*12))
ten_year_ago = (today - timedelta(weeks=52*10))
five_year_ago = (today - timedelta(weeks=52*5))
two_year_ago = (today - timedelta(days=365*2))
one_year_ago = (today - timedelta(days=365))
half_year_ago = (today - timedelta(days=183))
three_month_ago = (today - timedelta(days=90))
one_month_ago = (today - timedelta(days=30))
ten_day_ago = (today - timedelta(days=10))
five_day_ago = (today - timedelta(days=5))
yesterday = (today - timedelta(days=1))

fifteen_year_ago_s = stringify(fifteen_year_ago)
twelve_year_ago_s = stringify(twelve_year_ago)
ten_year_ago_s = stringify(ten_year_ago)
five_year_ago_s = stringify(five_year_ago)                            
two_year_ago_s = stringify(two_year_ago)
one_year_ago_s = stringify(one_year_ago)
half_year_ago_s = stringify(half_year_ago)
three_month_ago_s = stringify(three_month_ago)
one_month_ago_s = stringify(one_month_ago)
ten_day_ago_s = stringify(ten_day_ago)
five_day_ago_s = stringify(five_day_ago)
yesterday_s = stringify(yesterday)





class YahooFinance:
  query_url = 'https://query2.finance.yahoo.com/v10/finance' #/quoteSummary/AAPL?modules=%20defaultKeyStatistics'
  
  @classmethod
  def market_cap(cls, ticker):
#     assetProfile
#     financialData
#     defaultKeyStatistics
#     calendarEvents
#     incomeStatementHistory
#     cashflowStatementHistory
#     balanceSheetHistory

    url = cls.query_url + '/quoteSummary/{}?modules=defaultKeyStatistics'.format(ticker)
    
    data = None
    mkt_cap = None
    
    try:
      with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    except:
      pass
    
    if data is not None and data['quoteSummary']['error'] is None:
      enterpriseValue = data['quoteSummary']['result'][0]['defaultKeyStatistics']['enterpriseValue']
      if 'raw' in enterpriseValue.keys():
        mkt_cap = enterpriseValue['raw']
        try:
          mkt_cap = float(mkt_cap)
        except:
          pass

    return mkt_cap

  @classmethod
  def asset_profile(cls, ticker):
    url = cls.query_url + '/quoteSummary/{}?modules=assetProfile'.format(ticker)

    data = None

    try:
      with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    except Exception as ex:
      print(ex)
  
    if data is not None and data['quoteSummary']['error'] is None:
      data = data['quoteSummary']['result'][0]

    return data

  @classmethod
  def get_daily(cls, symbol, start_time=datetime(2000, 1, 3), end_time=today):
    df = web.DataReader(symbol, 'yahoo', start_time, end_time) 
    df.index = df.index.date.astype(str)
    df.rename(columns={'index': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adjusted_close'}, inplace=True)
    
    if 'adjusted_close' in df.columns:
      wanted_cols = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
    else:
      wanted_cols = ['open', 'high', 'low', 'close', 'volume']
    
    df = df[wanted_cols]
    df['symbol'] = symbol
    df.index.name = 'timestamp'

    return df


class Stooq:
  @classmethod
  def get_daily(cls, symbol, start_time=datetime(2010, 1, 1), end_time=today):

    if symbol == '^GSPC':
      stooq_symbol = '^SPX'
    elif symbol == '^IXIC':
      stooq_symbol = '^NDQ'
    else:
      stooq_symbol = symbol

    df = web.DataReader(stooq_symbol, 'stooq', start_time, end_time)   # given in desc order
    df.sort_index(inplace=True)
    df.index = df.index.date.astype(str)

    df.rename(columns={'index': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adjusted_close'}, inplace=True)

    df['adjusted_close'] = df['close']
    
    if 'adjusted_close' in df.columns:
      wanted_cols = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
    else:
      wanted_cols = ['open', 'high', 'low', 'close', 'volume']

    df = df[wanted_cols]
    df['symbol'] = symbol
    df.index.name = 'timestamp'

    return df


class AlphaVantage:
  query_url = 'https://www.alphavantage.co/query'
  api_key = ALPHAVANTAGE_API_KEY
  
  @classmethod
  def symbol_search(cls, keywords):
    url = cls.query_url + '?function=SYMBOL_SEARCH&keywords={}&apikey={}'.format(keywords, cls.api_key)
    
    with urllib.request.urlopen(url) as url:
      data = json.loads(url.read().decode())
      
#     [match for match in data['bestMatches'] if match['1. symbol'] == 'BA']
    return data

  @classmethod
  def get_last_refreshed_df(cls, metadata):
    '''
    Input:

    Series metadata

    Output:

    DataFrame with symbol and last refreshed date as columns
    '''
    last_refreshed_df = pd.DataFrame([(k, v['3. Last Refreshed']) for k, v in metadata.items()], columns=['symbol', 'last_refresh'])
    last_refreshed_df.set_index('symbol', inplace=True)

    return last_refreshed_df

  @classmethod
  def batch_stock_quotes(cls, symbols):
    if not isinstance(symbols, list):
      symbols = [symbols]
      
    url = cls.query_url + '?function=BATCH_STOCK_QUOTES&symbols={}&apikey={}'.format(','.join(symbols), cls.api_key)
    with urllib.request.urlopen(url) as url:
      data = json.loads(url.read().decode())
      
      #quotes = pd.DataFrame(data['Stock Quotes'])
      #quotes.rename(columns={'1. symbol': 'symbol', '2. price': 'price', '3. volume': 'volume', '4. timestamp': 'timestamp'}, inplace=True)
            
    #return quotes
    return data

  @classmethod
  def get_daily(cls, 
                symbols, 
                df_filename, 
                metadata_filename, 
                adjusted=True, 
                series_type=None, 
                save_every=100, 
                wait_sec_at_save=60., 
                output_size='compact', 
                api_key=None, 
                use_temp_file=False):
    '''
    if symbols is a list of symbols (not tuple), then get the daily adjusted time series of the stock, signal, etc.
    if symbols is a list of 2-tuple, it is interpreted as (from_symbol, to_symbol), then get the daily forex 

    Inputs:

    symbols: list of symbols
    df_filename: name of dataframe file storing the time series daily adjusted data.
    metadata_filename: name of the corresponding metadata pickle file
    series_type: Optional, valid values: [SECURITY, FOREX, CRYPTO, INDEX]. Indicate the type of the series. Note if this is crypto, this needs to be explicit 
    output_size: Optional, valid values: [compact, full]. If compact, only the latest 100 data points are returned. If full, all available data is returned.

    Output: 

    err_symbol_list: list of symbols that failed.
    '''

    if use_temp_file:
      temp_file = tempfile.NamedTemporaryFile(delete=False)
      temp_filename = temp_file.name  # store the temporary filename so you can load it later
      temp_file.close()  # close the file, we'll use it later
      print(f'Using temp_filename: {temp_filename}')
    else:
      temp_filename = None

    # try to guess at what series_type is based on the type(symbols)
    if series_type is None and isinstance(symbols, list) and len(symbols) > 0:
      if isinstance(symbols[0], str):
        series_type = 'SECURITY'
      elif isinstance(symbols[0], tuple):
        series_type = 'FOREX'
      else:
        pass

    if os.path.exists(df_filename):
      time_series_daily_df = pd.read_feather(df_filename)
    else:
      time_series_daily_df = None

    if os.path.exists(metadata_filename):
      last_metadata = load_from_pickle(metadata_filename)
    else:
      last_metadata = {}

    #metadata = {} if is_new_run else last_metadata.copy()
    metadata = {}
    
    err_symbol_list = []
    dfs = []
    for k, symbol in enumerate(symbols):
      if series_type == 'FOREX':
        from_symbol, to_symbol = symbol
      elif series_type == 'CRYPTO':
        _symbol, market = symbol

      if symbol in metadata.keys():     # can skip in case if this is a continuition run 
        continue

      print('symbol: {}'.format(symbol))
      
      try:
        if series_type == 'SECURITY':
          try:         
            stock = Stock(symbol=symbol, api_key=api_key)
            if adjusted:
              _df, _metadata = stock.get_daily_adjusted(output_size=output_size)    # this has become premium service since late 2021
            else:              
              _df, _metadata = stock.get_daily(output_size=output_size)        # this has become premuim service around late 2022
          except Exception as e:
            print(e)
            try: #yahoo
              # print("Try yahoo")
              # _df = YahooFinance.get_daily(symbol=symbol)
              # _metadata = {'1. Information': 'Daily Prices (open, high, low, close) and Volumes', 
              #     '2. Symbol': symbol, '3. Last Refreshed': today_s, '4. Output Size': 'Full size', '5. Time Zone': 'US/Eastern'}
              # print("Yahoo is successful.")
              # stooq
              print("Try Stooq")
              _df = Stooq.get_daily(symbol=symbol)
              _metadata = {'1. Information': 'Daily Prices (open, high, low, close) and Volumes', 
                  '2. Symbol': symbol, '3. Last Refreshed': today_s, '4. Output Size': 'Full size', '5. Time Zone': 'US/Eastern'}
              print("Stooq is successful.")

            except Exception as e:
              print(e)
              raise RemoteDataError(f"{symbol}: No data fetched using Stooq. Failed to get proper data in dataframe and metadata (a dict)")
              
        elif series_type == 'FOREX':
          forex = Forex(from_symbol=from_symbol, to_symbol=to_symbol, api_key=api_key)
          _df, _metadata = forex.get_daily()
        elif series_type == 'CRYPTO':
          crypto = Crypto(symbol=_symbol, market=market, api_key=api_key)
          _df, _metadata = crypto.get_daily()
        elif series_type == 'INDEX':
          if symbol == '^VIX':
            _df = CBOE.get_daily()
            _metadata = {'1. Information': 'Daily Prices (open, high, low, close) and Volumes', 
                    '2. Symbol': symbol, '3. Last Refreshed': today_s, '4. Output Size': 'Full size', '5. Time Zone': 'US/Eastern'}

          else:
            # _df = YahooFinance.get_daily(symbol=symbol)      # Nasdaq (the index) is not available on Alphavantage, get from Yahoo
            _df = Stooq.get_daily(symbol=symbol)
            _metadata = {'1. Information': 'Daily Prices (open, high, low, close) and Volumes', 
                    '2. Symbol': symbol, '3. Last Refreshed': today_s, '4. Output Size': 'Full size', '5. Time Zone': 'US/Eastern'}
            # else:
            #   index = Stock(symbol=symbol)                       # TODO: Consider renaming this class
            #   _df, _metadata = index.get_daily()
        else:
          pass

        dfs.append(_df)
        metadata[symbol] = _metadata
      except Exception as e:
        # print("Error: {}".format(symbol))
        print(e)
        err_symbol_list.append(symbol)

      wait_sec = np.random.random()*0.1 + 12.3   # this is roughly the right throttle for AlphaVantage 
      time.sleep(wait_sec)

      if k % save_every == 0:      # save and take a 60s break every N times
        print("Pause to save:")
        if len(dfs) == 0:
          continue
        df = pd.concat(dfs, axis=0)
        df.reset_index(inplace=True)       # timestamp was the index
        if series_type == 'SECURITY' or series_type == 'INDEX':
          print('# of symbols: {}, len of df: {}'.format(df.symbol.nunique(), len(df)))
        elif series_type == 'FOREX':
          print('# of symbols: {}, len of df: {}'.format((df.from_symbol+df.to_symbol).nunique(), len(df)))
        elif series_type == 'CRYPTO':
          print('# of symbols: {}, len of df: {}'.format((df.symbol+df.market).nunique(), len(df)))
        else:
          pass

        if use_temp_file:          
          if Path(temp_filename).stat().st_size > 0:
             # If the temp file has data, read it, concat with current df, and save back to temp file
            temp_df = pd.read_feather(temp_filename)            
            df = pd.concat([temp_df, df], axis=0, ignore_index=True)

          df.reset_index(drop=True).to_feather(temp_filename)
        else:
          # combine with whats done before          
          time_series_daily_df = pd.concat([time_series_daily_df, df], axis=0, ignore_index=True)

          # Dedup and keep only the latest 
          if series_type == 'SECURITY' or series_type == 'INDEX':
            time_series_daily_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last', inplace=True)
          elif series_type == 'FOREX':
            time_series_daily_df.drop_duplicates(subset=['timestamp', 'from_symbol', 'to_symbol'], keep='last', inplace=True)
          elif series_type == 'CRYPTO':
            time_series_daily_df.drop_duplicates(subset=['timestamp', 'symbol', 'market'], keep='last', inplace=True)  
          else:
            pass

          time_series_daily_df.reset_index(drop=True).to_feather(df_filename)  # 'datatime' was index

        dfs = []
        gc.collect()

        # merge with last_metadata 
        merged_metadata = {**last_metadata, **metadata}
        save_to_pickle(merged_metadata, metadata_filename)

        time.sleep(wait_sec_at_save)   # sleep at every save

    # Final save (remainder) 
    if len(dfs) > 0:
      df = pd.concat(dfs, axis=0)
      df.reset_index(inplace=True)       # timestamp was the index
      if use_temp_file:
        if Path(temp_filename).stat().st_size > 0:
          temp_df = pd.read_feather(temp_filename)
          df = pd.concat([temp_df, df], axis=0, ignore_index=True)

        time_series_daily_df = pd.concat([time_series_daily_df, df], axis=0, ignore_index=True)
        # os.remove(temp_filename)  # remove the temporary file   # TODO: uncomment this after debug

      else:
        time_series_daily_df = pd.concat([time_series_daily_df, df], axis=0, ignore_index=True)
      # Dedup 
      if series_type == 'SECURITY' or series_type == 'INDEX':
        time_series_daily_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last', inplace=True)
      elif series_type == 'FOREX':
        time_series_daily_df.drop_duplicates(subset=['timestamp', 'from_symbol', 'to_symbol'], keep='last', inplace=True)
      elif series_type == 'CRYPTO':
        time_series_daily_df.drop_duplicates(subset=['timestamp', 'symbol', 'market'], keep='last', inplace=True)

      time_series_daily_df.reset_index(drop=True).to_feather(df_filename)

      merged_metadata = {**last_metadata, **metadata}
      save_to_pickle(merged_metadata, metadata_filename)

    return err_symbol_list


  @classmethod
  def get_company_overview(cls, symbols, df_filename, save_every=100, wait_sec_at_save=60., api_key=None):

    '''
    Get company overview for symbols

    Output: 

    err_symbol_list: list of symbols that failed.
    '''
    def merge_and_save_df_to_disk():
      nonlocal overview_df
      df = pd.concat(dfs, axis=0)
      print('# of symbols: {}, len of df: {}'.format(df.Symbol.nunique(), len(df)))

      # combine with what's done before
      df.reset_index(inplace=True)      # timestamp was the index
      overview_df = pd.concat([overview_df, df], axis=0, ignore_index=True)

      # dedup and keep only the latest
      overview_df.drop_duplicates(subset=['timestamp', 'Symbol'], keep='last', inplace=True)

      # save to disk
      overview_df.reset_index(drop=True).to_feather(df_filename)   # 'datetime' was index


    overview_df = pd.read_feather(df_filename) if Path(df_filename).exists() else None

    err_symbol_list = []
    dfs = []

    for k, symbol in enumerate(symbols):
      print(f'symbol: {symbol}')

      try:
        co = Company(symbol=symbol, api_key=api_key)        
        _df = co.get_overview()

        dfs.append(_df)

      except Exception as e:
        print(e)
        err_symbol_list.append(symbol)
        #TODO: # try: # yahoo

      wait_sec = np.random.random()*0.1 + 15.0   # this is roughly the right throttle for AlphaVantage 
      time.sleep(wait_sec)
      
      if k % save_every == 0:      # save and take a 60s break every N times 
        print("Pause to save:")
        if len(dfs) == 0: continue

        # print(f'overview_df: {overview_df}')
        merge_and_save_df_to_disk()

        dfs = []    # reset for next round
        gc.collect()     

        time.sleep(wait_sec_at_save)

    # Final save (if there's remainder)
    if len(dfs) > 0:
      merge_and_save_df_to_disk()

    return err_symbol_list


class Forex:
  def __init__(self, from_symbol, to_symbol, api_key=None):
    self.from_symbol = from_symbol
    self.to_symbol = to_symbol
    self.api_key = api_key if api_key is not None else AlphaVantage.api_key

    self.query_url = 'https://www.alphavantage.co/query'

  def get_daily(self):
    url = self.query_url + '?function=FX_DAILY&from_symbol={}&to_symbol={}&outputsize=full&apikey={}'.format(self.from_symbol, self.to_symbol, self.api_key)
    with urllib.request.urlopen(url) as url:
      data = json.loads(url.read().decode())
      metadata = data['Meta Data']
      ts_df = pd.DataFrame(data['Time Series FX (Daily)']).T
      ts_df.sort_index(inplace=True)

      ts_df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close'}, inplace=True)
      ts_df = ts_df.astype('float')

      ts_df['from_symbol'] = self.from_symbol
      ts_df['to_symbol'] = self.to_symbol

      ts_df.index.name = 'timestamp'

    return ts_df, metadata

class Crypto:
  def __init__(self, symbol, market, api_key=None):
    self.symbol = symbol
    self.market = market
    self.api_key = api_key if api_key is not None else AlphaVantage.api_key

    self.query_url = 'https://www.alphavantage.co/query'

  def get_daily(self):
    url = self.query_url + '?function=DIGITAL_CURRENCY_DAILY&symbol={}&market={}&apikey={}'.format(self.symbol, self.market, self.api_key)
    with urllib.request.urlopen(url) as url:
      data = json.loads(url.read().decode())
      metadata = data['Meta Data']
      ts_df = pd.DataFrame(data['Time Series (Digital Currency Daily)']).T
      ts_df.sort_index(inplace=True)      # index is timestamp

      col_regex = re.compile(r'.*?\.\s+(.*?)$')   # "1a. open (CNY)" -> open (CNY)

      ts_df.rename(columns={col: col_regex.match(col).group(1) for col in ts_df.columns}, inplace=True)
      ts_df = ts_df.astype('float')

      ts_df.index.name = 'timestamp'

      # select only open, high, low, close for the given market (don't include USD)
      selected_cols = ['{} ({})'.format(x, self.market) for x in ['open', 'high', 'low', 'close']] + ['volume', 'market cap (USD)']
      ts_df = ts_df[selected_cols]

      # drop 'market cap', this is a bug, it is always the same as volume 
      ts_df.drop(columns=['market cap (USD)'], axis=1, inplace=True)

      # further rename to reduce redundancies
      col_regex_2 = re.compile(r'(.*?)\s+\({}\)'.format(self.market))   # close (CAD) -> close
      rename_col_dict = {col: col_regex_2.match(col).group(1) for col in ts_df.columns if col_regex_2.match(col) is not None}
      ts_df.rename(columns=rename_col_dict, inplace=True)

      # if the market is USD, then need to dedup some columns
      ts_df = ts_df.loc[:, ~ts_df.columns.duplicated()]

      ts_df['symbol'] = self.symbol
      ts_df['market'] = self.market

    return ts_df, metadata

class Stock:
  def __init__(self, symbol, api_key=None):
    self.symbol = symbol
    if api_key is None:
      self.api_key = AlphaVantage.api_key
    else:
      self.api_key = api_key
      
    self.query_url = 'https://www.alphavantage.co/query'

  def get_quote(self, verbose=False):
    url = self.query_url + '?function=GLOBAL_QUOTE&symbol={}&apikey={}'.format(self.symbol, self.api_key)

    with urllib.request.urlopen(url) as url:
      data = json.loads(url.read().decode())
      if verbose:
        print(data)

      quote = data['Global Quote']

      key_regex = re.compile(r'.*?\.\s+(.*?)$')        # "01. open" -> open

      ts_df = pd.DataFrame({key_regex.match(k).group(1): v for k, v in quote.items()}, index=[datetime.now()])
      ts_df.index.name = 'timestamp'

      return ts_df

  def get_daily_adjusted(self, output_size='compact', verbose=False):
    data = self._fetch_data('TIME_SERIES_DAILY_ADJUSTED', {'outputsize': output_size})
    if verbose: print(data)
    ts_df = self._to_dataframe(data, 'Time Series (Daily)')
    self._rename_columns(ts_df, r'.*?\.\s+(.*?)$')
    metadata = data['Meta Data']
    return ts_df, metadata


  def get_intraday(self, interval='1min', verbose=False):
      """
      Alphavantage supports interval: 1min, 5min, 15min, 30min, 60min
      """
      
      data = self._fetch_data('TIME_SERIES_INTRADAY', {'interval': interval})
      if verbose:
          print(data)
      ts_df = self._to_dataframe(data, f'Time Series ({interval})')
      self._rename_columns(ts_df, r'.*?\.\s+(.*?)$')
      metadata = data['Meta Data']
      return ts_df, metadata


  def get_daily(self, output_size='compact', verbose=False):
    data = self._fetch_data('TIME_SERIES_DAILY', {'outputsize': output_size})
    if verbose:
        print(data)
    ts_df = self._to_dataframe(data, 'Time Series (Daily)')
    self._rename_columns(ts_df, r'.*?\.\s+(.*?)$')
    metadata = data['Meta Data']
    return ts_df, metadata

  
  def _fetch_data(self, function, params):
    params['function'] = function   # url params
    params['symbol'] = self.symbol
    params['apikey'] = self.api_key
    url = self.query_url + "?" + "&".join([f"{k}={v}" for k, v in params.items()])
    try:
      with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
        if 'Error Message' in data:
          specific_error_msg = data["Error Message"]
          raise Exception(specific_error_msg)
        return data
    except Exception as e:
      # raise RemoteDataError(f"{self.symbol}: Error fetching data using AlphaVantage.")
      raise RemoteDataError(f"{self.symbol}: Error fetching data using AlphaVantage. {specific_error_msg if specific_error_msg else ''}")
    
  def _to_dataframe(self, data, series_key):
    ts_df = pd.DataFrame(data[series_key]).T
    ts_df.sort_index(inplace=True)
    ts_df = ts_df.astype('float')
    ts_df['symbol'] = self.symbol
    ts_df.index.name = 'timestamp'
    return ts_df
  
  def _rename_columns(self, df, pattern):
    col_regex = re.compile(pattern)
    
    # Ensure that we only rename columns if they match the regex pattern
    renamed_columns = {col: col_regex.match(col).group(1) 
                       for col in df.columns 
                       if col_regex.match(col)}
    
    df.rename(columns=renamed_columns, inplace=True)





class Company:
  def __init__(self, symbol, api_key=None):
    self.symbol = symbol
    if api_key is None:
      self.api_key = AlphaVantage.api_key
    else:
      self.api_key = api_key
      
    self.query_url = 'https://www.alphavantage.co/query'

  def get_overview(self, verbose=False):
    url = self.query_url + f'?function=OVERVIEW&symbol={self.symbol}&apikey={self.api_key}'

    try:
      with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
        if verbose: print(data)

        df = pd.DataFrame(data=data, index=[datetime.today().date()])
        df.index.name = 'timestamp'

        df.Symbol   # if Symbol is absent in df, this will throw ex

    except Exception as e:
      raise RemoteDataError(f"{self.symbol}: No data fetched using Avantage. Failed to get overview.")
      return None

    return df

class PutCallRatio:
  # Data from https://www.theocc.com/webapps/historical-volume-query
  years = ['2016', '2017', '2018', '2019']
  months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
  mth_idx_2_str = {i+1:m for i, m in enumerate(months)}

  columns = ['timestamp',
          'equity_calls', 'equity_puts', 'equity_pc_ratio',
          'index_calls', 'index_puts', 'index_pc_ratio',
          'debt_calls', 'debt_puts', 'debt_pc_ratio',
          'occ_calls', 'occ_puts', 'occ_total', 'occ_pc_ratio'
  ]
  
  @classmethod
  def dataframe(cls, data_path):
    pc_ratio_df = []
    
    #### Past years
    for yr in cls.years:
      for mth in cls.months:
        _pc_ratio_df = pd.read_csv(data_path/'pc_ratio_{}_{}.txt'.format(mth, yr), sep='\t', index_col=False)

        _pc_ratio_df.columns = cls.columns
        _pc_ratio_df.timestamp = _pc_ratio_df.timestamp + '/{}'.format(yr)
        _pc_ratio_df.sort_values(by='timestamp', ascending=True, inplace=True)
        _pc_ratio_df.set_index('timestamp', inplace=True)

        pc_ratio_df.append(_pc_ratio_df)

    ##### This year
    this_yr = str(datetime.today().year)
    all_mths_so_far = [cls.mth_idx_2_str[m+1] for m in range(datetime.today().month)]

    for mth in all_mths_so_far:
      _pc_ratio_df = pd.read_csv(data_path/'pc_ratio_{}_{}.txt'.format(mth, this_yr), sep='\t', index_col=False)

      _pc_ratio_df.columns = cls.columns
      _pc_ratio_df.timestamp = _pc_ratio_df.timestamp + '/{}'.format(this_yr)
      _pc_ratio_df.sort_values(by='timestamp', ascending=True, inplace=True)
      _pc_ratio_df.set_index('timestamp', inplace=True)

      pc_ratio_df.append(_pc_ratio_df)

    pc_ratio_df = pd.concat(pc_ratio_df, axis=0)    

    pc_ratio_df.index = pd.to_datetime(pc_ratio_df.index, yearfirst=True)
    pc_ratio_df.index = pc_ratio_df.index.date
    pc_ratio_df.index.name = 'timestamp'

    # fix bad data,  pc ratio should not be 0.0, use use fill forward
    pc_ratio_df.equity_pc_ratio.replace(to_replace=0, method='ffill', inplace=True)
    pc_ratio_df.index_pc_ratio.replace(to_replace=0, method='ffill', inplace=True)
    pc_ratio_df.occ_pc_ratio.replace(to_replace=0, method='ffill', inplace=True )

    return pc_ratio_df

class CBOE:
  @classmethod
  def get_daily(cls):
    resp = requests.get('https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv')
    df = pd.read_csv(StringIO(resp.text))

    df.DATE = pd.to_datetime(df.DATE)
    df.index = df.DATE
    df.drop(columns=['DATE'], inplace=True)
    df.sort_index(inplace=True)
    df.index = df.index.date.astype(str)
    df.index.name = 'timestamp'

    df.rename(columns={'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close'}, inplace=True)

    df['adjusted_close'] = df['close']

    df['symbol'] = '^VIX'

    return df

    

def plot_ma(s_df):
  '''
  df: a dataframe containing time series of one stock
  '''
  symbol = s_df.symbol.unique()[0]
  s_df.index = pd.to_datetime(s_df.index) 

  s_px = s_df.adjusted_close
  ma50 = s_px.rolling(window=50).mean()
  ma100 = s_px.rolling(window=100).mean()
  ma200 = s_px.rolling(window=200).mean()

  print("last date: {}".format(s_df.iloc[-1].name))

  ax = s_px.loc[one_year_ago:].plot(label='{}'.format(symbol), marker='*')
  ma50.loc[one_year_ago:].plot(label='MA50')
  ma100.loc[one_year_ago:].plot(label='MA100')
  ma200.loc[one_year_ago:].plot(label='MA200')

  # round to nearest month.
  datemin = s_px.loc[one_year_ago:].iloc[0:1].index.values[0]
  datemax = s_px.iloc[-1:].index + timedelta(days=15)
  ax.set_xlim(datemin, datemax)

  ax.xaxis.set_major_locator(mdates.MonthLocator())
  # ax.xaxis.set_major_formatter(fmt)

  # ax.grid(True)

  plt.legend();


class APIKeyManager:
  def __init__(self, key_store: Union[str, Path] = 'alphavantage_keys_df') -> None:
    key_store = Path(key_store)
    if not key_store.exists():
      raise FileNotFoundError(f"Error: key store '{key_store}' does not exist.")
    
    self.key_store = key_store  

  def get_available_key(self) -> str:
    '''
    Retrieve an available key. Will raise ValueError if no valid key is found.        
    '''
    self._read_keystore()

    # Iterate through rows and find a key that's available
    for _, row in self.api_keys_df.iterrows():
      # last_used_time = row['last_used']
      
      # if (pd.isna(last_used_time) or (datetime.now() - last_used_time) > timedelta(hours=24)) and not row['lock']:
      #   return row['value']
      if self.is_key_available(row['value']):
        return row['value']
        
    # If no available key is found
    raise ValueError("No available API key found.")

  def is_key_available(self, key) -> bool:
    '''
    Check if the given key is available. Returns True if the key is available, False otherwise.
    '''
    row = self.api_keys_df.query("value == @key")
    if len(row) > 0:
      row = row.iloc[0]
      last_used_time = row['last_used']
      usage_count = row['usage_count']
      
      # if (pd.isna(last_used_time) or (datetime.now() - last_used_time) > timedelta(hours=24)) and not row['lock'] and usage_count < 25:
      if not row['lock'] and usage_count < 25:
        return True
      else:
        return False

    # If the key is not found
    return False

  
  def record_key_last_used_timestamp(self, key, last_used=None) -> None:
      """
       Record the usage of the given API key. Optionally, a custom timestamp and usage details can be provided.

        :param key: The API key being used.
        :param last_used: Optional custom timestamp for when the key was used.
        :param script_name: Optional name of the script where the key was used.
        :param part_number: Optional part number associated with the script.
        :return: None
      """
      if last_used is None:
        last_used = datetime.now()  # right now (at time of call) should be default.

      current_ip = self.get_public_ip()

      # load the key store again (in case it was updated by another process)
      # this is a small file, so crossing finger that there's no corruption by 2 parties read/writing at
      # the same time, since we don't have high concurrent use cases.
      # TODO: fix concurrency to be super robust and not risking corrupting the key store
      self._read_keystore()
      
      # Update the last_used column for the corresponding key
      idx = self.api_keys_df[self.api_keys_df['value'] == key].index
      if len(idx) == 0:
        raise ValueError(f"No API key with value '{key}' found.")
      
      self.api_keys_df.loc[idx, 'last_used'] = last_used
      self.api_keys_df.loc[idx, 'last_ip_used'] = current_ip
      
      # Save the updated DataFrame back to the key store
      self.save()

  def inc_usage_count(self, key) -> None:
    self._read_keystore()
    idx = self.api_keys_df[self.api_keys_df['value'] == key].index
    if len(idx) == 0:
        raise ValueError(f"No API key with value '{key}' found.")

    self.api_keys_df.loc[idx, 'usage_count'] += 1
    self.save()

  def reset_usage_counts(self) -> None:
    """
    Reset the usage count of each key if the current time is more than a day since it was last used.
    
    :return: None
    """
    self._read_keystore()

    for idx, row in self.api_keys_df.iterrows():
      last_used_time = row['last_used']
      if pd.isna(last_used_time) or (datetime.now() - last_used_time) > timedelta(days=1):
        self.api_keys_df.at[idx, 'usage_count'] = 0

    self.save()

  def invalidate_key(self, key: str) -> None:
    """
    Invalidate a key purposely, so it won't be used for 24 hours.

    :param key: The API key to invalidate.
    :return: None
    """
    self._read_keystore()

    idx = self.api_keys_df[self.api_keys_df['value'] == key].index
    if len(idx) == 0:
      raise ValueError(f"No API key with value '{key}' found.")

    # Set the key's usage_count to 25 and last_used to the current time
    self.api_keys_df.loc[idx, 'usage_count'] = 25
    self.api_keys_df.loc[idx, 'last_used'] = datetime.now()

    self.save()
    
  def install_new_key(self, name: str, value: str) -> None:
    '''
    name is logical key name
    value is the actual key value (random looking string)
    '''
    self._read_keystore()
   
    new_key_row = pd.DataFrame([{'name': name, 'value': value, 'last_used': pd.NaT, 'lock': False, 'usage_count': 0}])
    self.api_keys_df = pd.concat([self.api_keys_df, new_key_row], axis=0, ignore_index=True)

    self.save()

  def remove_key(self, value: str) -> None:
    self._read_keystore()
    idx = self.api_keys_df[self.api_keys_df['value'] == value].index
    if len(idx) == 0:
      raise ValueError(f"No API key '{value}' found.")
    
    self.api_keys_df.drop(idx, inplace=True)
    self.api_keys_df.defrag_index(inplace=True)
    self.save()

  def lock(self, key) -> None:
    """ Lock a specific API key. """
    self._read_keystore()

    idx = self.api_keys_df[self.api_keys_df['value'] == key].index
    if len(idx) == 0:
        raise ValueError(f"No API key with value '{key}' found.")
    
    self.api_keys_df.loc[idx, 'lock'] = True
    self.save()

  def unlock(self, key) -> None:
    """ Unlock a specific API key. """
    self._read_keystore()

    idx = self.api_keys_df[self.api_keys_df['value'] == key].index
    if len(idx) == 0:
        raise ValueError(f"No API key with value '{key}' found.")
    
    self.api_keys_df.loc[idx, 'lock'] = False
    self.save()

  def unlock_all(self) -> None:
    """ Unlock all API keys. """
    self._read_keystore()
    self.api_keys_df['lock'] = False
    self.save()

  def save(self) -> None:
    self.api_keys_df.to_feather(self.key_store)
  
  def get_public_ip(self) -> str:
    """
    Retrieve the current public IP address.

    :return: The current public IP address as a string.
    """
    response = requests.get('https://httpbin.org/ip')
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()['origin']

  def is_new_ip_for_key(self, key: str) -> bool:
    """
    Check if the current public IP is different from the last IP used with the given key.

    :param key: The API key to check against.
    :return: True if the current IP is different, False otherwise.
    """
    current_ip = self.get_public_ip()
    row = self.api_keys_df.query("value == @key")
    if len(row) > 0:
      last_ip = row['last_ip_used'].iloc[0]
      return current_ip != last_ip

    return True  # If the key is not found, assume it's a new IP

  def _read_keystore(self) -> None:
    try:
      self.api_keys_df = pd.read_feather(self.key_store)
      # sanity check presence of columns
      assert set(self.api_keys_df.columns) == {'last_used', 'lock', 'name', 'usage_count', 'value', 'last_ip_used'}, "unexpected cols in key store df"

    except pd.errors.FeatherError as fe:
      raise ValueError(f"Error reading feather format from key store '{self.key_store}': {fe}")
    except Exception as e:        
      raise ValueError(f"Error: {e}\nFailed to read key store '{self.key_store}'.")
  