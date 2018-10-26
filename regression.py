import numpy as np 
import pandas as pd
import quandl
import math
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression

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
#print(df.head())

x = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
x = preprocessing.scale(x)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
