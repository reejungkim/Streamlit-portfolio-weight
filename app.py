# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import datetime as dt
import pandas as pd
#to get financial data
from pandas_datareader import data as pdr
#import yfinance as yf
import streamlit as st

date_start = '2020-01-01'
date_end = dt.datetime.now() 
  

SP100_tickers = pd.read_csv('/Users/reejungkim/Documents/Git/Heroku_deployment/Streamlit/testingstreamlit/S&P100 tickers.csv')
print(SP100_tickers['Symbol'])


st.write("""
# Simple Stock Price App
Shown are the stock closing price and volume of Google!
""")


# %%time

df = pd.DataFrame()

for i in SP100_tickers['Symbol']:
    symbol = SP100_tickers.loc[SP100_tickers['Symbol']==i]
    try:
        symbol_data = pdr.DataReader(i, 'yahoo', date_start, date_end).reset_index()
        #display(symbol_data)  
    except (KeyError, ValueError):  # the error could possibly occur when there's "." in stock name 
        symbol_data = pdr.DataReader(i.replace('.','-'), 'yahoo', date_start, date_end).reset_index()
        #symbol_data = pd.DataFrame()
        pass
    except:
        print(i + " - Error.")
        symbol_data = pd.DataFrame()
        pass
    single_table = pd.concat([symbol, symbol_data], axis=0, ignore_index=True) #axis=0 <- row. add frames by row and use fill down.
    single_table['Symbol'].ffill(inplace=True)
    df = df.append(single_table)
    
    
df = df.loc[df['Date'].notnull()]
df = df.reset_index(drop=True)   



st.line_chart(df.Close)
st.line_chart(df.Volume)




st.write("""
    | Price | Volume |
    | ------|:-------|

"""    )