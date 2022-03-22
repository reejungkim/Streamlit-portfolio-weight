# -*- coding: utf-8 -*-


import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#for portofolio optimization 
import scipy.optimize as sco
import scipy.interpolate as sci
#to get financial data
from pandas_datareader import data as pdr
#import yfinance as yf
import streamlit as st

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout='wide')

st.write("Optimal portfolio weight calculation based on Markowitz portfolio theory")

st.write("""
    | Please select minimum 3 to 4 tickers to get optimal weight properly |
    | ------|

"""    )


# TICKERS
file = 'https://raw.githubusercontent.com/reejungkim/Streamlit/master/S%26P100%20tickers.csv'
SP100_tickers = pd.read_csv(file,  error_bad_lines=False)

#tickers_selected = st.multiselect("Select ticker(s)", SP100_tickers.Symbol)
tickers_selected =['AAPL', 'AMZN', 'UPS']
tickers_df = pd.DataFrame (tickers_selected,columns=['ticker'])


# DATES
st.sidebar.title("Select date range")
today = dt.datetime.now()
start_date = st.sidebar.date_input('Start date', today - dt.timedelta(days=7))
end_date = st.sidebar.date_input('End date', today - dt.timedelta(days=1)) 

if start_date < end_date:
    st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.error('Error: End date must fall after start date')
    
options_dropdown = ['Close', 'High', 'Low', 'Open', 'Adj Close']
price_indicator = st.sidebar.selectbox('Choose indicator', options_dropdown)

st.sidebar.text('Enter the risk free rate in %')
riskfree_input = st.sidebar.number_input('Enter risk free rate (%): ' ,  0.00)


# FUNCTIONS
def statistics(weights):
    ''' Return portfolio statistics.
    
    Parameters
    ==========
    weights : array-like
    
    Returns
    =======
    pret : float  (portfolio return)
    pvol : float   (portfoliio volatility)
    pret / pvol : float      #sharp ratio (beta)

    '''
    riskfree = riskfree_input   #risk free rate .19% (1 Year Treasury Rate is at 0.19%)
    weights = np.array(weights)
    pret = np.sum(logChange.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(logChange.cov() * 252, weights)))
    return np.array([pret, pvol, ((pret-riskfree )/ pvol)])

def min_func_sharpe(weights):
    return -statistics(weights)[2]

def min_func_variance(weights):
    return statistics(weights)[1] ** 2

def min_func_port(weights):
    return statistics(weights)[1]

def f(x):
    ''' efficient frontier (spline) '''
    return sci.splev(x, tck, der=0)

def f_derivative(x):
    ''' efficient frontier (first derivative)'''
    return sci.splev(x, tck, der=1)


def equations(p, rf = riskfree_input):
    eq1 = ( rf - p[0] )
    eq2 = ( rf + p[1] * p[2] - f(p[2]) )
    eq3 = p[1] - f_derivative(p[2])
    return eq1, eq2, eq3




if( tickers_selected != [] ):
# LOAD DATA
    df = pd.DataFrame()
    for i in tickers_selected:
        symbol = tickers_df.loc[tickers_df['ticker']==i]
        try:
            symbol_data = pdr.DataReader(i, 'yahoo', start_date, end_date)[price_indicator].reset_index()
            #display(symbol_data)  
        except (KeyError, ValueError):  # the error could possibly occur when there's "." in stock name 
            symbol_data = pdr.DataReader(i.replace('.','-'), 'yahoo', start_date, end_date)[price_indicator()].reset_index()
            #symbol_data = pd.DataFrame()
            pass
        except:
            print(i + " - Error.")
            symbol_data = pd.DataFrame()
            pass
        single_table = pd.concat([symbol, symbol_data], axis=0, ignore_index=True) #axis=0 <- row. add frames by row and use fill down.
        single_table['ticker'].ffill(inplace=True)
        #single_table = single_table.loc[single_table['Date'].notnull()]
        df = df.append(single_table) 

    df = df.loc[df[price_indicator].notnull()]
    st.write(df)
    #df = df.loc[df['Date'].notnull()]
    df = df.set_index('Date')  
# GRAPH
    # st.line_chart(df[price_indicator].groupby('Date').sum() )
    #st.line_chart(df.Volume)
    
    df = df.reset_index(drop=False)
    d = df.pivot_table(values=price_indicator , index='Date', columns='ticker', aggfunc=np.sum, margins=False)
    
    #variance (rate of change)
    logChange = np.log(d / d.shift(1)) 
    
    #expected annual return
    avg_return = pd.DataFrame(logChange.mean()*252*100 )
    #avg_return.rename( columns={0: 'expected return'}, inplace=True) 
    avg_return.columns = ['Average return (%)']
    st.write('expected annual return based on selected range of historical data', avg_return)

# SIMULATION
    prets = []  #stores list of portfolio returns
    pvols = []  #stores list of portolfio volatilities

    for p in range (5000):
        weights = np.random.random(len(tickers_selected))
        weights /= np.sum(weights)
        prets.append(np.sum(logChange.mean() * weights) * 252)
        pvols.append(np.sqrt(np.dot(weights.T, 
                            np.dot(logChange.cov() * 252, weights))))
    prets = np.array(prets)
    pvols = np.array(pvols)
    plt.scatter(pvols, prets, c=prets/pvols, marker='o', cmap='RdYlBu')
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')
    
    
# HIGHEST SHARP RATIO
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    bnds = tuple((0, 1) for x in range(len(tickers_selected)))
    opts = sco.minimize(min_func_sharpe, len(tickers_selected) * [1. / len(tickers_selected),], method='SLSQP', bounds=bnds, constraints=cons)
    
    optimal_weights_sharp = pd.DataFrame(opts['x']*100)
    optimal_weights_sharp.columns = ['Weight']
    optimal_weights_sharp.index = avg_return.index
    st.write('Weight of portfolio that returns highest sharp ratio', optimal_weights_sharp)
    pt_opts = statistics(opts['x']).round(3)
    plt.scatter(pt_opts[1], pt_opts[0], marker="*", s=500, alpha=0.5, color='black')

    st.write("exp return :" + str( statistics(opts['x'].round(3))[0].round(3) ) )
    st.write("exp volatility :" + str( statistics(opts['x'].round(3))[1].round(3) )  )
    st.write("exp sharp ratio :" + str( statistics(opts['x'].round(3))[2].round(3) )  )
    
# MINIMUM VARIANCE
    optv = sco.minimize(min_func_variance,  len(tickers_selected) * [1. / len(tickers_selected),], method='SLSQP',
                       bounds=bnds, constraints=cons)        
    # Optimal (minimum) volatility
    pt_optv = statistics(optv['x']).round(3)
    plt.plot(pt_optv[1], pt_optv[0], marker="*", markersize=20, alpha=0.5, color='black')

# EFFICIENT FRONTIER
    #cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},{'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    bnds = tuple((0, 1) for x in weights)
    
    trets = np.linspace( prets.min() , max(prets.max(), pt_opts[0].max()), 50)
    tvols = []
    for tret in trets:       #getting weight where minimum volatility occurs for each returned value
        cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
                {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        res = sco.minimize(min_func_port, len(tickers_selected)* [1. / len(tickers_selected),], method='SLSQP', 
                           bounds=bnds, constraints=cons)
        tvols.append(res['fun'])
    tvols = np.array(tvols)

    # Minimum variance frontier
    plt.scatter(tvols, trets, c=trets / tvols, marker='x', s=70, linewidth=2, cmap='plasma')
    
  
    # Efficient frontier
    ind = np.argmin(tvols)
    evols = tvols[ind:]    #include array only up to the index of minimum volatility
    erets = trets[ind:]
    tck = sci.splrep(evols, erets)
    plt.plot(evols, f(evols), lw=8, alpha=0.4, color='green')  
    
    
    opt = sco.fsolve(equations,[ riskfree_input, pvols.max()/2, prets.max()/2])
    #opt_weight = pd.DataFrame(opt)
    #opt_weight.columns = ['Optimal weight (%)']
    #opt_weight.index = avg_return.index


    #Capital market line
    #cx = np.linspace(0, 0.4)   # (0.0, 0.3)
    #plt.plot(cx, opt[2] + opt[2] * cx, lw=3, alpha=0.2, ) 
        
        
    #optimal portfolio  
    plt.plot(opt[2], f(opt[2]), 'r*', markersize=25.0, color='red') 


    
    cons =({'type':'eq','fun': lambda x: statistics(x)[0]- tret },
        {'type':'eq','fun': lambda x: np.sum(x)-1})
    result = sco.minimize(min_func_port,
                               len(tickers_selected)* [1. / len(tickers_selected),],
                               method = 'SLSQP',
                               bounds = bnds,
                               constraints = cons)
    
    optimal_weights = result['x'].round(3)
    
    portfolio = list(zip(avg_return.index, list(optimal_weights)))
    st.write('Optimal portfolio weight')
    st.write(pd.DataFrame(portfolio)) 
    
   
st.pyplot(plt)



#st.write("""
#    | Price | Volume |
#    | ------|:-------|

#"""    )