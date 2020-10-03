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

st.write("""
# Simple Stock Price App
Shown are the stock closing price and volume of S&P 100 stocks!
""")


# TICKERS
file = 'https://raw.githubusercontent.com/reejungkim/Streamlit/master/S%26P100%20tickers.csv'
SP100_tickers = pd.read_csv(file,  error_bad_lines=False)

tickers_selected = st.multiselect("Select ticker(s)", SP100_tickers.Symbol)
tickers_df = pd.DataFrame (tickers_selected,columns=['ticker'])


# DATES
st.sidebar.title("Select date range")
today = dt.datetime.now()
start_date = st.sidebar.date_input('Start date', today - dt.timedelta(days=30))
end_date = st.sidebar.date_input('End date', today - dt.timedelta(days=1)) 

if start_date < end_date:
    st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.error('Error: End date must fall after start date')


st.write(tickers_selected)

if( tickers_selected != [] ):
# LOAD DATA
    df = pd.DataFrame()
    for i in tickers_selected:
        symbol = tickers_df.loc[tickers_df['ticker']==i]
        try:
            symbol_data = pdr.DataReader(i, 'yahoo', start_date, end_date).reset_index()
            #display(symbol_data)  
        except (KeyError, ValueError):  # the error could possibly occur when there's "." in stock name 
            symbol_data = pdr.DataReader(i.replace('.','-'), 'yahoo', start_date, end_date).reset_index()
            #symbol_data = pd.DataFrame()
            pass
        except:
            print(i + " - Error.")
            symbol_data = pd.DataFrame()
            pass
        single_table = pd.concat([symbol, symbol_data], axis=0, ignore_index=True) #axis=0 <- row. add frames by row and use fill down.
        single_table['ticker'].ffill(inplace=True)
        df = df.append(single_table)     
    df = df.loc[df['Date'].notnull()]
    df = df.reset_index(drop=True)  
    st.line_chart(df.Close)
    #st.line_chart(df.Volume)



x = st.slider('Select a value')
st.write(x, 'squared is', x * x)


st.write("""
    | Price | Volume |
    | ------|:-------|

"""    )