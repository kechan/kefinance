from typing import List
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import time

from pandas.tseries.holiday import USFederalHolidayCalendar, AbstractHolidayCalendar,  GoodFriday, Holiday, nearest_workday, USMemorialDay, USLaborDay
from pandas.tseries.offsets import CustomBusinessDay

from .kefinance import Stock, APIKeyManager
from .common.utils import join_df

class Portfolio(ABC):
  def __init__(self, 
                txn_csv_path, 
                key_store_path,
                alphavantage_hist_quote_csv_path: Path,
                yahoo_hist_quote_csv_path: Path,
                yf_infos_df_path: Path,
                account_currency):
    self.key_manager = APIKeyManager(key_store=key_store_path)
    self.key_manager.reset_usage_counts()

    self.alphavantage_hist_quote_csv_path = alphavantage_hist_quote_csv_path
    self.yahoo_hist_quote_csv_path = yahoo_hist_quote_csv_path

    self.yf_infos_df_path = yf_infos_df_path

    self.wait_sec = np.random.random()*0.1 + 12.3   # this is roughly the right throttle for AlphaVantage API

    transactions_df = pd.read_csv(txn_csv_path, index_col=False, parse_dates=['Settlement Date', 'Transaction Date'])
    transactions_df = transactions_df.iloc[::-1].reset_index(drop=True)

    # rename columns to lowercase and replace space with underscore
    transactions_df.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

    self.transactions_df = transactions_df[transactions_df.account_currency == account_currency].copy()
    
    # some cleanup 
    self.cleanup_empty_symbol()
    self.fix_symbol_name_changes()

    if not self.transactions_df.q("symbol == ' ' and quantity != 0").empty:
      print("*** Warning ***: there are still empty symbols after cleanup. Please fix them.")

    # non US symbols may need preprocessing. e.g. TD -> TD.TO for canadian symbols
    self.preprocess_symbols()

    self.all_symbols = [s for s in self.transactions_df.symbol.unique() if s != ' ']
    print(f'# of all symbols involved in transactions: {len(self.all_symbols)}')

    # datetime related
    self.min_date = None    # to be computed later
    self.max_date = None    # to be computed later

    # historical quotes
    self.historic_quote_df = None   # to be computed later
    self.yf_historic_quote_df = None   # to be computed later

    # yahoo finance info
    self.yf_infos_df = None    

    self.title = 'Portfolio'

  @abstractmethod
  def cleanup_empty_symbol(self):
    raise NotImplementedError("subclass must implement this method")

  @abstractmethod
  def fix_symbol_name_changes(self):
    raise NotImplementedError("subclass must implement this method")

  def sort_by_txn_date(self):
    self.transactions_df = self.transactions_df.set_index('transaction_date').sort_index()

  def gen_min_max_date(self):
    self.min_date = self.transactions_df.index.min()
    self.max_date = datetime.now()

  def save(self, path):
    self.transactions_df.to_csv(path, index=False)

  def load(self, path):
    self.transactions_df = pd.read_csv(path, index_col=False, parse_dates=['Settlement Date', 'Transaction Date'])

  # market value and cash flow calculation
  @abstractmethod
  def construct_historic_quote_df(self):
    raise NotImplementedError("subclass must implement this method")

  def reset_historic_quote_df(self):
    self.historic_quote_df = None
    self.yf_historic_quote_df = None


  @abstractmethod
  def get_quotes(self, a_date: datetime.date, symbols: List[str]) -> pd.DataFrame:
    raise NotImplementedError("subclass must implement this method")

  def get_markey_days(self):
    # get market trading days up to today
    market_days = pd.to_datetime(self.historic_quote_df.q("@self.min_date <= timestamp and timestamp <= @self.max_date").sort_values(by='timestamp').timestamp.unique())
    print(f'total # of market trading days: {len(market_days)}')

    return market_days


  def gen_daily_mkt_value(self) -> None:
    # get and combine historic quotes
    # historic_quote_df = pd.read_csv(self.alphavantage_hist_quote_csv_path, index_col=0, parse_dates=True)
    # historic_quote_df.reset_index(inplace=True)
    # yf_historic_quote_df = pd.read_csv(self.yahoo_hist_quote_csv_path, index_col=0, parse_dates=True)
    # yf_historic_quote_df.reset_index(inplace=True)

    # combine alpha and yf quote
    # historic_quote_df = pd.concat([historic_quote_df, yf_historic_quote_df[historic_quote_df.columns]], axis=0)
    # historic_quote_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first', inplace=True)

    # get market trading days up to today
    self.construct_historic_quote_df()
    market_days = self.get_markey_days()

    # market_days = pd.to_datetime(yf_historic_quote_df.q("@self.min_date <= timestamp and timestamp <= @self.max_date").sort_values(by='timestamp').timestamp.unique())
    # print(f'total # of market trading days: {len(market_days)}')

    # compute the market value and cash flow for each of the trading days.

    # injection/withdrawal of cash
    daily_external_cash_flow = self.transactions_df.q("(settlement_amount > 0 or settlement_amount < 0) and (type == 'TRANSFER' or type == 'DEPOSIT')").reset_index().groupby('transaction_date').settlement_amount.sum()

    total_mkt_values = []
    for i, dt in enumerate(market_days):
      print(f'{dt.date()}...', end='' if i % 10 != 0 else '\n')
      _transaction_df = self.transactions_df.loc[:dt.date()]   # accumulate everything up to that specific date 
      portfolio_position = _transaction_df.groupby('symbol').agg({'quantity': 'sum'}).reset_index()
      portfolio_position.drop(index=portfolio_position.q("symbol == ' ' or quantity == 0.0").index, inplace=True)

      symbols = portfolio_position.symbol.tolist()
      # quotes_on_date = historic_quote_df.q("timestamp == @dt.date() and symbol.isin(@symbols)")
      quotes_on_date = self.get_quotes(dt.date(), symbols)
      stocks_with_quotes = join_df(portfolio_position, quotes_on_date, left_on='symbol', how='left')

      stocks_with_quotes['mkt_value'] = stocks_with_quotes.quantity * stocks_with_quotes.close
      mkt_value = stocks_with_quotes.mkt_value.sum()

      # cash deposit/withdrawal, or from buying/selling stocks, and dividends
      cumulative_settlement = _transaction_df['settlement_amount'].sum() 

      total_mkt_value = mkt_value + cumulative_settlement
      total_mkt_values.append(total_mkt_value)

    total_mkt_value_df = pd.DataFrame({'date': market_days, 'total_mkt_value': total_mkt_values})
    total_mkt_value_df = join_df(total_mkt_value_df, daily_external_cash_flow.rename('cash_flow').reset_index(), left_on='date', right_on='transaction_date', how='left')
    total_mkt_value_df.drop(columns=['transaction_date'], inplace=True)
    total_mkt_value_df.set_index('date', inplace=True)
    total_mkt_value_df.fillna(0, inplace=True)

    self.total_mkt_value_df = total_mkt_value_df
    self.daily_external_cash_flow = daily_external_cash_flow

    # current portfolio info
    current_portfolio = {} 
    current_portfolio['position'] = portfolio_position.copy()
    current_portfolio['stocks_quotes'] = stocks_with_quotes.copy()
    current_portfolio['mkt_value'] = mkt_value
    current_portfolio['cum_settlement'] = cumulative_settlement
    current_portfolio['total_mkt_value'] = total_mkt_value

    self.current_portfolio = current_portfolio


  # get historic quote from external source online

  def get_ignore_symbols_for_alphavantage(self) -> list:
    return []

  def get_ignore_symbols_for_yahoo(self) -> list:
    return []
  
  def get_historic_quote_from_alphavantage(self, batch_size=25, output_size='compact'):
    print(f'Getting hist quotes from alpha vantage for {len(self.all_symbols)} symbols...')

    for i in range(0, len(self.all_symbols), batch_size):
      end_index = min(i + batch_size, len(self.all_symbols))
      current_key = self._get_historic_quote_from_alphavantage(i, end_index, output_size=output_size)
      print(f'current_key: {current_key}')

      # Pause for IP change using NordVPN
      print(f"Batch {i // batch_size + 1} completed. Please change the IP before continuing.")
      input("Press Enter after changing the IP...")

    self.key_manager.invalidate_key(current_key)

    # since more historic quotes are added, we need to reset the historic quote df
    self.reset_historic_quote_df()

  def _get_historic_quote_from_alphavantage(self, start_index, end_index, output_size='compact') -> str:
    """
    Get historic quote from alphavantage for all symbols seen 
    Process a batch of symbols from start_index to end_index.

    :param start_index: Starting index of the batch.
    :param end_index: Ending index of the batch.

    :return: current alphavantage key used
    """
    skippable_symbols = self.get_ignore_symbols_for_alphavantage()

    historic_quote_dfs = []
    for symbol in self.all_symbols[start_index: end_index]:
      print(f'Processing {symbol}...')
      if symbol in skippable_symbols:
        print(f'Skipping {symbol}...')
        continue

      try:
        api_key = self.key_manager.get_available_key()
        self.key_manager.record_key_last_used_timestamp(key=api_key)
        print(f'api_key: {api_key}')
        stock = Stock(symbol=symbol, api_key=api_key)
        self.key_manager.inc_usage_count(key=api_key)

        df, metadata = stock.get_daily(verbose=False, output_size=output_size)
        historic_quote_dfs.append(df)
      except Exception as e:
        print(e)
      finally:
        time.sleep(self.wait_sec)

    historic_quote_df = pd.concat(historic_quote_dfs, axis=0)

    # merge new quote and save
    if self.alphavantage_hist_quote_csv_path.exists():
      print("Merging new quote with existing quote...")
      orig_historic_quote_df = pd.read_csv(self.alphavantage_hist_quote_csv_path, index_col=0, parse_dates=True)
      historic_quote_df = pd.concat([orig_historic_quote_df, historic_quote_df], axis=0)
    else:
      print(f"Creating and saving new historic quote file {self.alphavantage_hist_quote_csv_path} ...")

    historic_quote_df.reset_index(inplace=True)
    historic_quote_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first', inplace=True)
    historic_quote_df.to_csv(self.alphavantage_hist_quote_csv_path, index=False)

    return api_key

  def get_historic_quote_from_yahoo(self):
    historic_quote_dfs = []
    if self.min_date is None or self.max_date is None:
      self.sort_by_txn_date()   # assign txn date as index (warning: a side effect)
      self.gen_min_max_date()

    skippable_symbols = self.get_ignore_symbols_for_yahoo()

    for symbol in self.all_symbols:
      print(f'Processing {symbol}...')
      if symbol in skippable_symbols:
        print(f'Skipping {symbol}...')
        continue

      try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=self.min_date.date().strftime('%Y-%m-%d'), end=self.max_date.date().strftime('%Y-%m-%d'), auto_adjust=False)
        # remove tz and convert timestamp to date only
        df.index = df.index.tz_localize(None).date

        df.index.name = 'timestamp'
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df['symbol'] = symbol

        historic_quote_dfs.append(df)
      except Exception as e:
        print(e)
      finally:
        time.sleep(2.0)

    historic_quote_df = pd.concat(historic_quote_dfs, axis=0)

    # merge new quote and save
    if self.yahoo_hist_quote_csv_path.exists():
      print("Merging new quote with existing quote...")
      orig_historic_quote_df = pd.read_csv(self.yahoo_hist_quote_csv_path, index_col=0, parse_dates=True)
      historic_quote_df = pd.concat([orig_historic_quote_df, historic_quote_df], axis=0)
    else:
      print(f"Creating and saving new historic quote file {self.yahoo_hist_quote_csv_path} ...")

    historic_quote_df.reset_index(inplace=True)
    historic_quote_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first', inplace=True)
    historic_quote_df.to_csv(self.yahoo_hist_quote_csv_path, index=False)

    # since more historic quotes are added, we need to reset the historic quote df
    self.reset_historic_quote_df()

  # get sector, industry, and profile from external source (yfinance)
  def get_sector_industry_from_yfinance(self):
    """
    Get sector and industry info from yfinance for all symbols in my current portfolio
    """
    all_current_symbols = self.current_portfolio['position'].symbol.tolist()

    if self.yf_infos_df_path.exists():
      yf_infos_df = pd.read_feather(self.yf_infos_df_path)
    else:
      yf_infos_df = pd.DataFrame()

    new_yf_infos = []
    for symbol in all_current_symbols:
      if yf_infos_df.empty or not symbol in yf_infos_df.symbol.unique():
        print(f'Processing {symbol}...')
        new_yf_infos.append(yf.Ticker(symbol).info)
        time.sleep(1.0)

    if len(new_yf_infos) > 0:
      new_yf_infos_df = pd.DataFrame(new_yf_infos)
      yf_infos_df = pd.concat([yf_infos_df, new_yf_infos_df], axis=0, ignore_index=True)

      def assign_etf_sector_and_industry(row):
        if row['quoteType'] == 'ETF':
          row['sector'] = 'ETF'  # Assign a value for the sector
          row['industry'] = 'ETF'  # Assign a value for the industry
        return row

      yf_infos_df = yf_infos_df.apply(assign_etf_sector_and_industry, axis=1)  # a bit mem inefficient, but ok for now
      yf_infos_df.to_feather(self.yf_infos_df_path)

    self.yf_infos_df = yf_infos_df

  # generate current portfolio position with sector info
  def gen_current_portfolio_positions(self, columns=None):
    if columns is None:
      columns = ['symbol', 'sector', 'industry', 'longBusinessSummary']
      
    if self.yf_infos_df is None:
      self.get_sector_industry_from_yfinance()     # will populate self.yf_infos_df

    if 'dividendYield' in self.yf_infos_df.columns:
      columns.append('dividendYield')

    # print(columns)
    current_portfolio_position = join_df(self.current_portfolio['stocks_quotes'], self.yf_infos_df[columns], left_on='symbol', how='left')

    self.current_portfolio_position = current_portfolio_position

  # abstract method for symbol manipulation (relevant for CAD so far)
  @abstractmethod
  def preprocess_symbols(self):
    raise NotImplementedError("subclass must implement this method")

  def ytd_return(self, this_year_in_str: str):
    '''
    Not returning anything for now, just reporting for use in Notebook
    '''
    this_year = datetime.strptime(this_year_in_str, '%Y')
    last_year = this_year - timedelta(days=365)
    last_year_in_str = last_year.strftime('%Y')

    # get the last day of last year from total_mkt_value_df
    last_day_of_last_yr = self.total_mkt_value_df.loc[last_year_in_str].iloc[-1].name.date()
    last_day_of_last_yr_in_str = last_day_of_last_yr.strftime('%Y-%m-%d')

    this_yr_mkt_value_df = self.total_mkt_value_df.loc[last_day_of_last_yr_in_str:].copy()
    mkt_value_start = this_yr_mkt_value_df.iloc[0].total_mkt_value
    print(f'market value at year begin: ${mkt_value_start:.2f}')
    print(f"latest market value: ${self.total_mkt_value_df.iloc[-1].total_mkt_value:.2f}")

    this_yr_mkt_value_df['ytd_pct_gain'] = (this_yr_mkt_value_df.total_mkt_value - mkt_value_start)/mkt_value_start

    this_yr_mkt_value_df.ytd_pct_gain.plot(title=self.title, ylabel='ytd % gain');

    print(f'YTD gain %: {this_yr_mkt_value_df.ytd_pct_gain.iloc[-1]*100:.2f}%')

    tot_cash_flow_for_this_yr = 0.0
    try:
      tot_cash_flow_for_this_yr = self.daily_external_cash_flow[this_year_in_str].values.sum()
    except KeyError:
      tot_cash_flow_for_this_yr = 0.0

    if tot_cash_flow_for_this_yr != 0.0:
      cash_flow_adj_gain = (this_yr_mkt_value_df.iloc[-1].total_mkt_value - tot_cash_flow_for_this_yr - mkt_value_start)/mkt_value_start
      print(f'YTD gain % (cash flow adj): {cash_flow_adj_gain*100:.2f}%')

  def year_ago_return(self):
    latest_timestamp = self.total_mkt_value_df.iloc[-1].name
    # 1 yr ago from latest timestamp
    one_yr_ago = latest_timestamp - pd.DateOffset(years=1)

    one_yr_mkt_value_df = self.total_mkt_value_df.loc[one_yr_ago:].copy()
    mkt_value_one_year_ago = self.total_mkt_value_df.q("@one_yr_ago <= index").iloc[0].total_mkt_value
    print(f'market value 1 yr ago: ${mkt_value_one_year_ago:.2f}')
    print(f"latest market value: ${self.total_mkt_value_df.iloc[-1].total_mkt_value:.2f}")

    one_yr_mkt_value_df['pct_gain'] = (one_yr_mkt_value_df.total_mkt_value - mkt_value_one_year_ago)/mkt_value_one_year_ago

    one_yr_mkt_value_df.pct_gain.plot(title=self.title, ylabel='1 yr % gain');

    print(f'1 yr gain %: {one_yr_mkt_value_df.pct_gain.iloc[-1]*100:.2f}%')

    tot_cash_flow_last_one_year = 0.0
    try:
      tot_cash_flow_last_one_year = self.daily_external_cash_flow.loc[one_yr_ago:].values.sum()
    except KeyError:
      tot_cash_flow_last_one_year = 0.0

    if tot_cash_flow_last_one_year != 0.0:
      cash_flow_adj_gain = (one_yr_mkt_value_df.iloc[-1].total_mkt_value - tot_cash_flow_last_one_year - mkt_value_one_year_ago)/mkt_value_one_year_ago
      print(f'1 yr gain % (cash flow adj): {cash_flow_adj_gain*100:.2f}%')

    
    

  # for debug and review 
  def review_empty_symbol(self) -> pd.DataFrame:
    return self.transactions_df.q("symbol == ' ' and quantity != 0")


class USMarginPortfolio(Portfolio):
  def __init__(self, txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path):
    super().__init__(txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path, account_currency='USD')
    self.title = 'US Margin Portfolio'

  def preprocess_symbols(self):
    pass  # nothing to do for US margin account

  def get_ignore_symbols_for_alphavantage(self):
    return []

  def get_ignore_symbols_for_yahoo(self):
    return ['GMCR', 'PCLN', 'WFM', 'CBG', 'CELG', 'ACIA', 'TGE', 'MGP']

  def cleanup_empty_symbol(self):
    '''
    The code fix is the result of manual inspection of the transactions_df.
    This may need to be updated periodically.
    '''
    if len(self.transactions_df.q("symbol == ' ' and quantity != 0")) > 0:
      idx = self.transactions_df.q("description.str.contains('TALLGRASS') and symbol == ' '").index
      self.transactions_df.loc[idx, 'symbol'] = 'TGE'

      idx = self.transactions_df.q("description.str.contains('MGM GROWTH') and symbol == ' '").index
      self.transactions_df.loc[idx, 'symbol'] = 'MGP'

      idx = self.transactions_df.q("description.str.contains('POWERSHARES QQQ TRUST') and symbol == ' '").index
      self.transactions_df.loc[idx, 'symbol'] = 'QQQ'

      idx = self.transactions_df.q("description.str.contains('PRICELINE GROUP') and symbol == ' '").index
      self.transactions_df.loc[idx, 'symbol'] = 'PCLN'

      idx = self.transactions_df.q("description.str.contains('MONSTER BEVERAGE CORP') and symbol == ' '").index
      self.transactions_df.loc[idx, 'symbol'] = 'MNST'

      idx = self.transactions_df.q("description.str.contains('GOOGLE INC CL A') and symbol == ' '").index
      self.transactions_df.loc[idx, 'symbol'] = 'GOOGL'

    assert self.transactions_df.q("symbol == ' ' and quantity != 0").empty, 'need to fill in symbol'

  def fix_symbol_name_changes(self):
    self.transactions_df.loc[self.transactions_df.symbol == 'JP', 'symbol'] = 'JPPYY'

  def construct_historic_quote_df(self):
    # get and combine historic quotes
    if self.historic_quote_df is None or self.yf_historic_quote_df is None:
      self.historic_quote_df = pd.read_csv(self.alphavantage_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.historic_quote_df.reset_index(inplace=True)
      self.yf_historic_quote_df = pd.read_csv(self.yahoo_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.yf_historic_quote_df.reset_index(inplace=True)

      # combine alpha and yf quote
      self.historic_quote_df = pd.concat([self.historic_quote_df, self.yf_historic_quote_df[self.historic_quote_df.columns]], axis=0)
      self.historic_quote_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first', inplace=True)

  def get_quotes(self, a_date: datetime.date, symbols: List[str]) -> pd.DataFrame:
    return self.historic_quote_df.q("timestamp == @a_date and symbol.isin(@symbols)")


class CAMarginPortfolio(Portfolio):
  def __init__(self, txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path):
    super().__init__(txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path, account_currency='CAD')
    self.title = 'CA Margin Portfolio'

  def preprocess_symbols(self):
    self.transactions_df.symbol = self.transactions_df.symbol.str.replace('.', '-', regex=False)
    self.transactions_df.symbol = self.transactions_df.symbol.apply(lambda s: s + '.TO' if len(s) > 0 and s != ' ' else s)

  def cleanup_empty_symbol(self):
    '''
    The code fix is the result of manual inspection of the transactions_df.
    This may need to be updated periodically.
    '''
    if (len(self.transactions_df.q("symbol == ' ' and quantity != 0")) > 0):
      idx = self.transactions_df.q("description.str.contains('TRANSCANADA') and symbol == ' '").index
      self.transactions_df.loc[idx, 'symbol'] = 'TRP'

      idx = self.transactions_df.q("description.str.contains('MARIJUANA') and symbol == ' '").index
      self.transactions_df.loc[idx, 'symbol'] = 'HMMJ'

      idx = self.transactions_df.q("description.str.contains('CANADIAN PACIFIC RAILWAY') and symbol == ' '").index
      self.transactions_df.loc[idx, 'symbol'] = 'CP'

    assert self.transactions_df.q("symbol == ' ' and quantity != 0").empty, 'need to fill in symbol'

  def fix_symbol_name_changes(self):
    pass

  def get_ignore_symbols_for_alphavantage(self):
    return ['WMD.TO']

  def get_ignore_symbols_for_yahoo(self):
    return ['WMD.TO']

  def construct_historic_quote_df(self):
    # get historic quotes
    if self.historic_quote_df is None or self.yf_historic_quote_df is None:
      self.historic_quote_df = pd.read_csv(self.alphavantage_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.historic_quote_df.reset_index(inplace=True)

      self.yf_historic_quote_df = pd.read_csv(self.yahoo_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.yf_historic_quote_df.reset_index(inplace=True)

  def get_quotes(self, a_date: datetime.date, symbols: List[str]) -> pd.DataFrame:
    quotes_on_date = self.historic_quote_df.q("timestamp == @a_date and symbol.isin(@symbols)")
    cols = quotes_on_date.columns.tolist()

    # yahoo historic quote can't be totally trusted if there's a reverse split, it looks like Close isn't the original price but adjusted for the split.
    # we will consult yf_historic_quote_df only if its missing from alphavantage
    # check with yf_historic_quote_df
    yf_quotes_on_date = self.yf_historic_quote_df.q("timestamp == @a_date and symbol.isin(@symbols)")[cols]

    alpha_df = quotes_on_date.set_index(['timestamp', 'symbol'])
    yf_df = yf_quotes_on_date.set_index(['timestamp', 'symbol'])
    combined_quotes_df = alpha_df.combine_first(yf_df).reset_index()

    return combined_quotes_df


class USRRSPPortfolio(Portfolio):
  def __init__(self, txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path):
    super().__init__(txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path, account_currency='USD')
    self.title = 'US RRSP Portfolio'

  def preprocess_symbols(self):
    pass  # nothing to do for US RRSP account

  def cleanup_empty_symbol(self):
    pass

  def fix_symbol_name_changes(self):
    pass    

  def construct_historic_quote_df(self):
     # get and combine historic quotes, similar to USMarginPortfolio
    if self.historic_quote_df is None or self.yf_historic_quote_df is None:
      self.historic_quote_df = pd.read_csv(self.alphavantage_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.historic_quote_df.reset_index(inplace=True)
      self.yf_historic_quote_df = pd.read_csv(self.yahoo_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.yf_historic_quote_df.reset_index(inplace=True)

      # combine alpha and yf quote
      self.historic_quote_df = pd.concat([self.historic_quote_df, self.yf_historic_quote_df[self.historic_quote_df.columns]], axis=0)
      self.historic_quote_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first', inplace=True)

  def get_quotes(self, a_date: datetime.date, symbols: List[str]) -> pd.DataFrame:
    # similar to USMarginPortfolio
    return self.historic_quote_df.q("timestamp == @a_date and symbol.isin(@symbols)")


class CARRSPPortfolio(Portfolio):
  def __init__(self, txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path):
    super().__init__(txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path, account_currency='CAD')
    self.title = 'CA RRSP Portfolio'
  
  def preprocess_symbols(self):
    self.transactions_df.symbol = self.transactions_df.symbol.str.replace('.', '-', regex=False)
    self.transactions_df.symbol = self.transactions_df.symbol.apply(lambda s: s + '.TO' if len(s) > 0 and s != ' ' else s)

  def cleanup_empty_symbol(self):
    if len(self.transactions_df.q("symbol == ' ' and quantity != 0")) > 0:
      idx = self.transactions_df.q("description.str.contains('SILVER WHEATON') and symbol == ' '").index
      self.transactions_df.loc[idx, 'symbol'] = 'SLW'
      
      idx = self.transactions_df.q("description.str.contains('CANADIAN PACIFIC RAILWAY') and symbol == ' '").index
      self.transactions_df.loc[idx, 'symbol'] = 'CP'

    assert self.transactions_df.q("symbol == ' ' and quantity != 0").empty, 'need to fill in symbol'  

  def fix_symbol_name_changes(self):
    # OTC -> OTEX without any warning or notice
    idx = self.transactions_df.q("symbol == 'OTC'").index
    self.transactions_df.loc[idx, 'symbol'] = 'OTEX'

  def get_ignore_symbols_for_alphavantage(self):
    return ['SLW.TO']

  def get_ignore_symbols_for_yahoo(self):
    return ['SLW.TO']


  def construct_historic_quote_df(self):
    # get historic quotes
    if self.historic_quote_df is None or self.yf_historic_quote_df is None:
      self.historic_quote_df = pd.read_csv(self.alphavantage_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.historic_quote_df.reset_index(inplace=True)

      self.yf_historic_quote_df = pd.read_csv(self.yahoo_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.yf_historic_quote_df.reset_index(inplace=True)

  def get_quotes(self, a_date: datetime.date, symbols: List[str]) -> pd.DataFrame:
    quotes_on_date = self.historic_quote_df.q("timestamp == @a_date and symbol.isin(@symbols)")
    cols = quotes_on_date.columns.tolist()

    # yahoo historic quote can't be totally trusted if there's a reverse split, it looks like Close isn't the original price but adjusted for the split.
    # we will consult yf_historic_quote_df only if its missing from alphavantage
    # check with yf_historic_quote_df
    yf_quotes_on_date = self.yf_historic_quote_df.q("timestamp == @a_date and symbol.isin(@symbols)")[cols]

    alpha_df = quotes_on_date.set_index(['timestamp', 'symbol'])
    yf_df = yf_quotes_on_date.set_index(['timestamp', 'symbol'])
    combined_quotes_df = alpha_df.combine_first(yf_df).reset_index()

    return combined_quotes_df

class USTFSAPortfolio(Portfolio):
  def __init__(self, txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path):
    super().__init__(txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path, account_currency='USD')
    self.title = 'US TFSA Portfolio'
  
  def preprocess_symbols(self):
    pass  # nothing to do for US TFSA account

  def cleanup_empty_symbol(self):
    pass

  def fix_symbol_name_changes(self):
    pass

  def get_ignore_symbols_for_alphavantage(self):
    return ['TDB166', 'TDB2915', 'NBEV']

  def get_ignore_symbols_for_yahoo(self):
    return ['NBEV', 'TDB166', 'TDB2915']

  def construct_historic_quote_df(self):
     # get and combine historic quotes, similar to USMarginPortfolio
    if self.historic_quote_df is None or self.yf_historic_quote_df is None:
      self.historic_quote_df = pd.read_csv(self.alphavantage_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.historic_quote_df.reset_index(inplace=True)
      self.yf_historic_quote_df = pd.read_csv(self.yahoo_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.yf_historic_quote_df.reset_index(inplace=True)

      # combine alpha and yf quote
      self.historic_quote_df = pd.concat([self.historic_quote_df, self.yf_historic_quote_df[self.historic_quote_df.columns]], axis=0)
      self.historic_quote_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first', inplace=True)

  def get_quotes(self, a_date: datetime.date, symbols: List[str]) -> pd.DataFrame:
    # similar to USMarginPortfolio
    return self.historic_quote_df.q("timestamp == @a_date and symbol.isin(@symbols)")


class CATFSAPortfolio(Portfolio):
  def __init__(self, txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path):
    super().__init__(txn_csv_path, key_store_path, alphavantage_hist_quote_csv_path, yahoo_hist_quote_csv_path, yf_infos_df_path, account_currency='CAD')
    self.title = 'CA TFSA Portfolio'
 
  def preprocess_symbols(self):
    self.transactions_df.symbol = self.transactions_df.symbol.str.replace('.', '-', regex=False)
    self.transactions_df.symbol = self.transactions_df.symbol.apply(lambda s: s + '.TO' if len(s) > 0 and s != ' ' else s)

  def get_ignore_symbols_for_alphavantage(self):
    return ['HIVE.TO']

  def get_ignore_symbols_for_yahoo(self):
    return ['HIVE.TO']

  def cleanup_empty_symbol(self):
    pass

  def fix_symbol_name_changes(self):
    # reconcile the switch from QETH.U to QETH.UN
    # we will live with minor discrepancy due to diff currencies for now
    idx = self.transactions_df.q("symbol == 'QETH.U'").index
    self.transactions_df.loc[idx, 'symbol'] = 'QETH.UN'

  def construct_historic_quote_df(self):
    # get historic quotes
    if self.historic_quote_df is None or self.yf_historic_quote_df is None:
      self.historic_quote_df = pd.read_csv(self.alphavantage_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.historic_quote_df.reset_index(inplace=True)

      self.yf_historic_quote_df = pd.read_csv(self.yahoo_hist_quote_csv_path, index_col=0, parse_dates=True)
      self.yf_historic_quote_df.reset_index(inplace=True)

  def get_quotes(self, a_date: datetime.date, symbols: List[str]) -> pd.DataFrame:
    quotes_on_date = self.historic_quote_df.q("timestamp == @a_date and symbol.isin(@symbols)")
    cols = quotes_on_date.columns.tolist()

    # yahoo historic quote can't be totally trusted if there's a reverse split, it looks like Close isn't the original price but adjusted for the split.
    # we will consult yf_historic_quote_df only if its missing from alphavantage
    # check with yf_historic_quote_df
    yf_quotes_on_date = self.yf_historic_quote_df.q("timestamp == @a_date and symbol.isin(@symbols)")[cols]

    alpha_df = quotes_on_date.set_index(['timestamp', 'symbol'])
    yf_df = yf_quotes_on_date.set_index(['timestamp', 'symbol'])
    combined_quotes_df = alpha_df.combine_first(yf_df).reset_index()

    return combined_quotes_df