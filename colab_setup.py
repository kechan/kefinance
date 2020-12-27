import os, sys, time, pickle, pytz
from pathlib import *
from datetime import date, datetime, timedelta
from functools import partial

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import seaborn as sns
# sns.set(style='darkgrid', context='talk', palette='Dark2')
sns.set(rc={'figure.figsize': (11, 4)})
mpl.rc('figure', figsize=(16, 7))
style.use('ggplot')

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['font.size'] = 16

def onColab(): return os.path.exists('/content/')
# def onGCP(): return os.path.exists('/home/jupyter')
def onLocal(): return os.path.exists('/Users/kelvinchan')

bOnColab = onColab()
# bOnGCP = onGCP()
bOnLocal = onLocal()

if bOnColab:
  from google.colab import auth
  auth.authenticate_user()
  print('Authenticated')

if bOnColab and not os.path.exists('/content/drive'):   #presence of /content indicates you are on google colab
  from google.colab import drive
  drive.mount('/content/drive')
  print('gdrive mounted')


if bOnColab:
  home = Path('/content/drive/My Drive')
  data = home/'kefinance'/'data'
  tmp = home/'tmp'
elif bOnLocal:
  home = Path('/Users/kelvinchan/Google Drive')
  data = Path('/Users/kelvinchan/Documents/kefinance')
  tmp = Path('/tmp')
else:
  print("Unknown env")

finance_utils_path = home/'kefinance'/'utils'
sys.path.insert(0, str(finance_utils_path))

from kefinance import YahooFinance, AlphaVantage, Forex, Stock, Crypto, PutCallRatio
from kefinance import plot_ma

from common_util import load_from_pickle, save_to_pickle
from small_fastai_utils import join_df

# pre-defined relative dates
from kefinance import fifteen_year_ago, twelve_year_ago, ten_year_ago, five_year_ago, two_year_ago, one_year_ago, half_year_ago
from kefinance import three_month_ago, one_month_ago, 
from kefinance import five_day_ago, ten_day_ago, yesterday, today

from kefinance import fifteen_year_ago_s, twelve_year_ago_s, ten_year_ago_s, five_year_ago_s, two_year_ago_s, one_year_ago_s, half_year_ago_s
from kefinance import three_month_ago_s, one_month_ago_s, 
from kefinance import five_day_ago_s, ten_day_ago_s, yesterday_s, today_s

