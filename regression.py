import numpy as np 
import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL')
#print(df.head())
# adjusted: opening price, high, low, closing and volume of stock prices and trade
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
#print(df.head())
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

#math.ceil will round up to the nearsest whole number
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())