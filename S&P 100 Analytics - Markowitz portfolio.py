#!/usr/bin/env python
# coding: utf-8

# In[1]:


import timeit
import datetime as dt
import os
import subprocess 

import pandas as pd
import numpy as np

#for portofolio optimization 
import scipy.optimize as sco


#visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
#import plotly.graph_objs as go
get_ipython().run_line_magic('matplotlib', 'inline')
#visualization with seaborn
import seaborn as sns
#visualization with plotly 
import plotly.graph_objects as go #for candle chart visualization
import plotly.express as px #for html output


#to get financial data
from pandas_datareader import data as pdr


#for webscraping 
import requests
from bs4 import BeautifulSoup 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
import requests


#for pdf report generation
from reportlab.pdfgen import canvas
from reportlab.platypus import *
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus.tableofcontents import TableOfContents


# In[2]:


days_to_lookback =  20    #int (input('days to search financial data:'))


# In[3]:


riskfree = .0019


# In[4]:


date_end = dt.datetime.now() #.today()
date_start = date_end - dt.timedelta(days_to_lookback)


# # Create timestamped folder to save reports created later

# In[5]:


#timestamp = str(dt.datetime.now()) <- '2020-06-26 20:26:08.775766'  
timestamp =  dt.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
timestamp


# In[6]:


path = os.getcwd() 
reportDir = os.path.join(path, "Markowitz portflio output " + timestamp)
reportDir


# In[7]:


#create named folder
os.makedirs(reportDir, exist_ok = False)


# In[8]:


os.chdir(reportDir)


# # Web scraping S&P100 companies from Wikipedia table

# In[6]:


website_url = requests.get("https://en.wikipedia.org/wiki/S%26P_100").text

soup = BeautifulSoup(website_url,'lxml')
#print(soup.prettify())


# In[7]:


wiki_table = soup.find('table',{'class':'wikitable sortable'})


# In[9]:


data = []
tHeader = []
SP100_tickers = pd.DataFrame()

for tr in wiki_table.find_all("tr"):
    for th in tr.find_all("th")[:1]:
        tHeader = th.get_text(strip=True)
        SP100_tickers[tHeader] = []
    for td in tr.find_all("td")[:1]:
        data.append( td.get_text(strip=True))
        
SP100_tickers[tHeader] = data


# SP100_tickers.to_csv('S&P100 tickers.csv')

# # Use DataReader to get financial data of S&P100 companies from yahoo

# In[13]:


get_ipython().run_cell_magic('time', '', '\ndf = pd.DataFrame()\n\nfor i in data:\n    symbol = SP100_tickers.loc[SP100_tickers[\'Symbol\']==i]\n    try:\n        symbol_data = pdr.DataReader(i, \'yahoo\', date_start, date_end).reset_index()\n        #display(symbol_data)  \n    except (KeyError, ValueError):  # the error could possibly occur when there\'s "." in stock name \n        symbol_data = pdr.DataReader(i.replace(\'.\',\'-\'), \'yahoo\', date_start, date_end).reset_index()\n        #symbol_data = pd.DataFrame()\n        pass\n    except:\n        print(i + " - Error.")\n        symbol_data = pd.DataFrame()\n        pass\n    single_table = pd.concat([symbol, symbol_data], axis=0, ignore_index=True) #axis=0 <- row. add frames by row and use fill down.\n    single_table[\'Symbol\'].ffill(inplace=True)\n    df = df.append(single_table)\n    ')


# In[14]:


df = df.loc[df['Date'].notnull()]

df = df.reset_index(drop=True)


# df.groupby(['Symbol']).describe().transpose()

# In[16]:


df_Flat = df


# # Prepare data frame to display volatility and to be normalized

# In[17]:


d = df.pivot_table(values='Close', index='Date', columns='Symbol', aggfunc=np.sum, margins=False)


# In[18]:


d.head()


# In[19]:


logChange = np.log(d / d.shift(1)) # geometric brownian motion. - make the errors being normally distributed.
# more standardly used than pct_change.  df_temp['Close'] in this case.
logChange.head()


# # pick tickers to use for portfolio
# ( highest growth rates)

# In[20]:


tickers = pd.DataFrame((d.mean()/d.var()))
tickers.columns =['sharp']
tickers = tickers.sort_values(by='sharp', ascending=False).head(10)
symbols = tickers.index.tolist()
symbols


# tickers_bestPerformed = logChange.describe().transpose().sort_values(by='mean', ascending=False).head(5)
# symbols = tickers_bestPerformed.index.tolist()

# In[21]:


logChange = logChange[symbols]
logChange.head(3)


# ### expected return of each ticker picked

# In[22]:


logChange.mean()*252   #252 business days per year in general


# ##### covariance across tickers

# In[23]:


logChange.cov()*252


# In[24]:


sns.heatmap(logChange.cov()*252)


# # Portfolio with randomly assinged weights

# #### get random portofolio weights to run simulations

# In[25]:


np.random.seed(0)
weights = np.random.random(len(symbols))
weights /= np.sum(weights)
weights


# In[26]:


#expeced annual return of portfolio
np.sum(logChange.mean()*weights)*252


# In[27]:


#expected annual variance of portfolio
np.dot(weights.T, np.dot(logChange.cov() * 252, weights))


# In[28]:


#expected annual standard deviation of portfolio (volatility)
np.sqrt(np.dot(weights.T, np.dot(logChange.cov() * 252, weights)))


# # Simulation of portfolios with random weights

# In[29]:


prets = []  #stores list of portfolio returns
pvols = []  #stores list of portolfio volatilities

for p in range (5000):
    weights = np.random.random(len(symbols))
    weights /= np.sum(weights)
    prets.append(np.sum(logChange.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T, 
                        np.dot(logChange.cov() * 252, weights))))
prets = np.array(prets)
pvols = np.array(pvols)


# In[30]:



plt.scatter(pvols, prets, c=prets/pvols, marker='o', cmap='coolwarm') # mpl.cm.jet)

plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio') #beta
plt.show()


# In[31]:


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
    riskfree = .0019   #risk free rate .19% (1 Year Treasury Rate is at 0.19%)
    weights = np.array(weights)
    pret = np.sum(logChange.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(logChange.cov() * 252, weights)))
    return np.array([pret, pvol, ((pret-riskfree )/ pvol)])


# ### Sharp ratio optimization

# In[32]:


def min_func_sharpe(weights):
    return -statistics(weights)[2]


# In[33]:


cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
cons


# In[34]:


#return, volatility, and sharp ratio when portfolio is weighted equally
w = len(symbols)*[1. /len(symbols),]
statistics(w)


# In[35]:


bnds = tuple((0, 1) for x in range(len(symbols)))
bnds


# In[36]:


#set initial point
len(symbols)* [1./len(symbols),]


# #### Minimize a scalar function of one or more variables using Sequential Least Squares Programming (SLSQP).

# #### scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)

# In[37]:


get_ipython().run_cell_magic('time', '', "\nopts = sco.minimize(min_func_sharpe, len(symbols)* [1. / len(symbols),], method='SLSQP',\n                       bounds=bnds, constraints=cons)")


# In[38]:


opts


# In[39]:


a=opts['x'].round(3)   #opt weights
#print("exp return :" + str( np.sum(logChange.mean()*a)*252  )  ).round(3)   #1yr exp return of portfolio with opt weights 
print("exp return :" + str( statistics(opts['x'].round(3))[0].round(3) ) )
print("exp volatility :" + str( statistics(opts['x'].round(3))[1].round(3) )  )
print("exp sharp ratio :" + str( statistics(opts['x'].round(3))[2].round(3) )  )


# In[40]:


plt.scatter(pvols, prets, c=prets/pvols, marker='o', cmap='RdYlBu')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

#highest sharp ratio
pt_opts = statistics(opts['x']).round(3)
plt.scatter(pt_opts[1], pt_opts[0], marker="*", s=500, alpha=0.5, color='b')


#return, volatility, and sharp ratio when portfolio is weighted equally
w = len(symbols)*[1. /len(symbols),]
plt.scatter(statistics(w)[1], statistics(w)[0], marker="*", s= 500, alpha=0.5, color='g')


# add a red dot for maximum vol & maximum return
plt.scatter(pvols.max(), prets.max(), c='black', s=500, alpha=0.5, edgecolors='black')



plt.show()


# ### Volatility Optimization

# In[41]:


def min_func_variance(weights):
    return statistics(weights)[1] ** 2


# In[42]:


optv = sco.minimize(min_func_variance,  len(symbols) * [1. / len(symbols),], method='SLSQP',
                       bounds=bnds, constraints=cons)


# In[43]:


optv


# In[44]:


plt.scatter(pvols, prets, c=prets/pvols, marker='o', cmap='RdYlBu')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

#highest sharp ratio
pt_opts = statistics(opts['x']).round(3)
plt.scatter(pt_opts[1], pt_opts[0], marker="*", s=500, alpha=0.5, color='b')


# Optimal (minimum) volatility
pt_optv = statistics(optv['x']).round(3)
plt.plot(pt_optv[1], pt_optv[0], marker="*", markersize=20, alpha=0.5, color='r')
#plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0], 'y*', markersize=20, color='r')
           
    
plt.show()


# # Efficient Frontier

# ## Minimum variance frontier

# In[45]:


cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in weights)


# In[46]:


def min_func_port(weights):
    return statistics(weights)[1]


# In[47]:


prets.min(), prets.max(), pt_opts[0]


# In[48]:


get_ipython().run_cell_magic('time', '', "\n\ntrets = np.linspace( prets.min() , max(prets.max(), pt_opts[0].max()), 50)\ntvols = []\nfor tret in trets:       #getting weight where minimum volatility occurs for each returned value\n    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},\n            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})\n    res = sco.minimize(min_func_port, len(symbols)* [1. / len(symbols),], method='SLSQP',\n                       bounds=bnds, constraints=cons)\n    tvols.append(res['fun'])\ntvols = np.array(tvols)")


# In[49]:


tvols


# In[50]:


# rangdomly weighted sample portfolio simulations
plt.scatter(pvols, prets, c=prets / pvols, marker='o', cmap='RdYlBu')


# Minimum variance frontier
plt.scatter(tvols, trets, c=trets / tvols, marker='x', s=70, linewidth=2, cmap='plasma')
            
    
#highest sharp ratio
pt_opts = statistics(opts['x']).round(3)
plt.scatter(pt_opts[1], pt_opts[0], marker="*", s=500, alpha=0.5, color='b')


# Optimal (minimum) volatility
pt_optv = statistics(optv['x']).round(3)
plt.plot(pt_optv[1], pt_optv[0], marker="*", markersize=20, alpha=0.5, color='r')
#plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0], 'y*', markersize=20, color='r')

 
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()


# ## Efficient frontier

# In[51]:


import scipy.interpolate as sci


# In[52]:


np.argmin(tvols) #find index of minimum value of volatility 


# In[53]:


ind = np.argmin(tvols)
evols = tvols[ind:]    #include array only up to the index of minimum volatility
erets = trets[ind:]


# In[54]:


tck = sci.splrep(evols, erets)


# In[55]:


def f(x):
    ''' efficient frontier (spline) '''
    return sci.splev(x, tck, der=0)

def df(x):
    ''' efficient frontier (first derivative)'''
    return sci.splev(x, tck, der=1)
f(0)


# In[56]:


# rangdomly weighted sample portfolio simulations
plt.scatter(pvols, prets, c=prets / pvols, marker='o', cmap='RdYlBu')


# Efficient frontier
plt.plot(evols, f(evols), lw=8, alpha=0.4, color='green')  

# Minimum variance frontier
plt.scatter(tvols, trets, c=trets / tvols, marker='x', s=70, linewidth=2, cmap='plasma')
            
#highest sharp ratio
pt_opts = statistics(opts['x']).round(3)
plt.scatter(pt_opts[1], pt_opts[0], marker="*", s=500, alpha=0.5, color='b')

# Optimal (minimum) volatility
pt_optv = statistics(optv['x']).round(3)
plt.plot(pt_optv[1], pt_optv[0], marker="*", markersize=20, alpha=0.5, color='r')
#plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0], 'y*', markersize=20, color='r')
 

plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()


# ## Capital market line (CML)

# In[57]:


riskfree


# In[58]:


def equations(p, rf=riskfree):
    eq1 = rf-p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3


# In[59]:


pvols.max()/2, prets.max()/2


# In[60]:


opt = sco.fsolve(equations,[ riskfree, pvols.max()/2, prets.max()/2])
opt


# In[61]:


np.round(equations(opt), 8)


# In[62]:


# rangdomly weighted sample portfolio simulations
plt.scatter(pvols, prets, c=prets / pvols, marker='o', cmap='RdYlBu')


# Efficient frontier
plt.plot(evols, f(evols), lw=8, alpha=0.4, color='green')  

# Minimum variance frontier
plt.scatter(tvols, trets, c=trets / tvols, marker='x', s=70, linewidth=2, cmap='plasma')
            
#highest sharp ratio
pt_opts = statistics(opts['x']).round(3)
plt.scatter(pt_opts[1], pt_opts[0], marker="*", s=500, alpha=0.5, color='b')

# Optimal (minimum) volatility
pt_optv = statistics(optv['x']).round(3)
plt.plot(pt_optv[1], pt_optv[0], marker="*", markersize=20, alpha=0.5, color='r')
#plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0], 'y*', markersize=20, color='r')
 


 
#Capital market line
cx = np.linspace(riskfree, 0.3 )   # (0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, lw=3, alpha=0.2, ) 
    
    
#optimal portfolio  
plt.plot(opt[2], f(opt[2]), 'r*', markersize=25.0, color='black') 


plt.grid(True)
plt.axhline(0.0, color='k', ls='--', lw=2.0)
plt.axvline(0.0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()


# #### Optimal portfolio weights

# In[63]:


symbols


# In[64]:


cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - f(opt[2])},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
res = sco.minimize(min_func_port, len(symbols) * [1. / len(symbols),], method='SLSQP',
                       bounds=bnds, constraints=cons)
res['x'].round(3)


# In[65]:


statistics(res['x'].round(3))


# #### return, volatility, and sharp ratio expected with the optimal weights

# In[66]:


portfolio = list(zip(symbols, res['x'].round(3)))
portfolio


# In[67]:


def format_column_number(dataframe, listCols):
    for col in listCols:
        try:
            dataframe[col]= pd.Series([round(val, 2 ) for val in dataframe[col]], index= dataframe.index)
        except ValueError:  #skips error when the column is in format already
            dataframe[col] = dataframe[col]
            pass
    return dataframe

def format_column_percentage(dataframe, listCols):
    for col in listCols:
        try:
            dataframe[col] = pd.Series(["{0:.2f}%".format(val*100) for val in dataframe[col]], index= dataframe.index)
        except ValueError:  #skips error when the column is in format already
            dataframe[col] = dataframe[col]
        
    return dataframe


# In[68]:


summary_output = pd.DataFrame({'Symbol': symbols})
summary_output['Weight'] = res['x'].round(3)

summary_output = pd.merge(summary_output , (logChange.mean()*252).reset_index(name='temp'))

summary_output['Exp return'] = summary_output['Weight']*summary_output['temp']

summary_output = summary_output.pivot_table(index='Symbol', values=['Weight', 'Exp return']  , aggfunc=np.sum, margins=True)


format_column_percentage(summary_output, ['Exp return', 'Weight'])
summary_output


# In[69]:


today = dt.datetime.today().strftime('%Y-%m-%d')
pdf_file_name = "Report "+ today +".pdf"


# In[70]:


def PrepareTableForReportLab(dataframe):
    
    data = dataframe

    'handling grouped first index column'
    a= []
    for i in range(len(data)):
        #print(i, data.index.get_level_values(0)[i])
        if data.index.get_level_values(0)[i-1] == data.index.get_level_values(0)[i]:
            a.append(i)
    a.sort(reverse=True)

    as_list = data.index.get_level_values(0).tolist()
    for i in a:
        as_list[i] = ""

        
    'flat dataframe'
    data = data.reset_index()
    'remove duplicated values from first column which was index before resetting index'
    first_col_name = data.columns[0]
    data[first_col_name] = as_list 
    
    #data = dataframe.reset_index()
    #colwidths = 800/len(data.columns) 
    data = [data.columns.to_list()] + data.values.tolist() 

    #tbl = Table(data) # 
    tbl = Table(data) #, colwidths ) #, rowheights)
    tbl.setStyle(TableStyle([
    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
    ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
    ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
    ('BACKGROUND', (0,0), (-1,0), colors.Color(0,0.7,0.7))
    ]))
    
    return tbl


# In[71]:


story = []


# In[72]:


story.append(Paragraph("Portfolio", getSampleStyleSheet()['Heading1']))
story.append(PrepareTableForReportLab(summary_output))

story.append(PageBreak())


# In[73]:


doc = SimpleDocTemplate(pdf_file_name, pagesize = landscape(letter), topMargin = inch * .25, bottomMargin = inch * .25)
doc.build(story)


# In[74]:


#Open pdf file generated
subprocess.run(['open', pdf_file_name], check=True)


# In[ ]:




