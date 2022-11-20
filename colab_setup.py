import os, sys, time, pickle, pytz, argparse, gc
from pathlib import Path
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

bOnColab = Path('/content').exists()
bOnLocal = Path('/Users/kelvinchan').exists()

description = 'Script to setup custom ML environment for Colab, Kaggle, GCP VM, and local machine.'

parser = argparse.ArgumentParser(description=description)

# argument for email address
parser.add_argument('--gdrive_email_addr', type=str, help='gdrive address')

args = parser.parse_args()

gdrive_email_addr = args.gdrive_email_addr

if bOnColab:
  from google.colab import auth
  auth.authenticate_user()
  print('Authenticated')

if bOnColab and not os.path.exists('/content/drive'):   #presence of /content indicates you are on google colab
  from google.colab import drive
  drive.mount('/content/drive')
  print('gdrive mounted')


if bOnColab:
  home = Path('/content/drive/MyDrive')
elif bOnLocal:
  home = Path(f'/Users/kelvinchan/{gdrive_email_addr} - Google Drive/My Drive')
else:
  print("Unknown env")

data = home/'kefinance'/'data'
tmp = home/'tmp'

# if len([p for p in sys.path if 'kefinance/utils' in p]) == 0: 
#   sys.path.insert(0, str(home/'kefinance'/'utils'))

if len([p for p in sys.path if 'kefinance/kefinance' in p]) == 0: 
  sys.path.insert(0, str(home/'kefinance'/'kefinance'))


from kefinance import *
#from kefinance import YahooFinance, AlphaVantage, Forex, Stock, Crypto, PutCallRatio
#from kefinance import plot_ma

from kefinance.common.utils import load_from_pickle, save_to_pickle, join_df
# from common_util import load_from_pickle, save_to_pickle
# from small_fastai_utils import join_df

# pre-defined relative dates
from kefinance import fifteen_year_ago, twelve_year_ago, ten_year_ago, five_year_ago, two_year_ago, one_year_ago, half_year_ago
from kefinance import three_month_ago, one_month_ago 
from kefinance import five_day_ago, ten_day_ago, yesterday, today

from kefinance import fifteen_year_ago_s, twelve_year_ago_s, ten_year_ago_s, five_year_ago_s, two_year_ago_s, one_year_ago_s, half_year_ago_s
from kefinance import three_month_ago_s, one_month_ago_s 
from kefinance import five_day_ago_s, ten_day_ago_s, yesterday_s, today_s

