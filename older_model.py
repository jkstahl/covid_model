from scipy.integrate import odeint
import numpy as np 
from matplotlib import pyplot as plt 

class SIR:
    def __init__(self, N, TI0):
        '''
        TI0 is the total infected including recovered and infected.
        '''
        self.N = N
        self.TI0 = TI0
        print (self.N)
        print(self.TI0)
    
    def deriv (self, y , t, N, b, gamma, c):
        S, I, R = y
        dSdt = -(b) * S * I / (N - c * t)
        dIdt = (b)* S * I / (N - c * t) - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
        
    def calc_sir(self, t, R0, b, gamma, c, printit = False, return_sir = False):
        '''
        Need to find R0 since we only have infected + recovered
        '''
        N_new = self.N
        S0, I0, R0 = self.N - self.TI0, self.TI0 - R0, R0
        y0 = S0, I0, R0
        ret = odeint(self.deriv, y0, t, args=(N_new, b, gamma, c))
        ti = ret[:, 1] + ret[:, 2]
        if printit:
            print (ti)
            plt.plot (t, ti)
            plt.show()
        else:
            if return_sir:
                return ret
            else:
                return ti

import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

old_model = [ 5.35652036e+02,  1.10252107e+00,  2.94664807e-01, -2.51892865e+07]
old_model2 = [582.999990603501,66.39372548939426,0.0978685826779643,-4552930172.88384,]
old_model3 = (583,[582.9999948184237,170.0885952456969,0.07128952017223418,-13816365390.463554,])

START_INDEX = 50
ADDITIONAL_ROWS =1

sns.set(rc={'figure.figsize':(15, 7)})
#df = pd.read_csv(r'C:\Users\neoba\Downloads\time_series_covid19_confirmed_global.csv')
df = pd.read_csv(r'C:\Users\neoba\Downloads\time_series_covid19_confirmed_global (2).csv')
df = pd.read_csv(r'C:\Users\neoba\Downloads\time_series_covid19_confirmed_global (3).csv')
tests = pd.read_csv(r'C:\Users\neoba\Downloads\full-list-total-tests-for-covid-19.csv')
df = pd.read_csv(r'C:\Users\neoba\Downloads\time_series_covid19_confirmed_global (5).csv')
df = pd.read_csv(r'C:\Users\neoba\Downloads\time_series_covid19_confirmed_global (6).csv')
df = pd.read_csv(r'C:\Users\neoba\Downloads\time_series_covid19_confirmed_global (8).csv')
df = pd.read_csv(r'C:\Users\neoba\Downloads\time_series_covid19_confirmed_global (9).csv')
#df = pd.read_csv(r'C:\Users\neoba\Downloads\time_series_covid19_deaths_global.csv')
tests = tests.loc[tests['Code'] == 'USA']
tests['Date'] = pd.to_datetime(tests['Date'])
tests = tests.set_index('Date')
#print (tests)


df = df.loc[df['Province/State'].isnull()]
#df.rename(columns={'Country/Region':'Date'}, inplace=True)
#df['Date'] = pd.to_datetime(df['Date'])
df.rename(columns={'Country/Region':'Date'}, inplace=True)
df = df.set_index('Date')
df = df.T

df_infections =  df.iloc[START_INDEX:, :]
df_infections.index = pd.to_datetime(df_infections.index)

nd = df_infections.iloc[1:, :]
us_infections = nd - df_infections.iloc[:-1, :].set_index(nd.index)
#us_infections['US'].plot()
us_norm = us_infections['US'] / tests['Total tests'] 
#print (df_infections)
#print (df)



def create_spread_plot(country, df, extra_rows=ADDITIONAL_ROWS, model = None, log_scale=False, populations = {'US':300e6}):
    plt.figure()
    

    us_data = df.loc[:, [country]]

    #print (us_data.index)
    #popt, pcov = curve_fit(func, , , [100,400,0.001,0])
    y = us_data[country].astype(float).to_numpy()
    x = np.array(range(len(y)))
    
    if model != None:
        initial, model = model
    else:
        initial =y[0] 
    
    sir = SIR(float(populations[country]), initial)
    
    if model != None:
        popt = model
    else: 
        popt, pcov = curve_fit(sir.calc_sir, x, y, p0=(1, .2, 1./10, 1./10), maxfev=20000)
    

    
    print ('(%d ,[' % initial, end='')
    for p in popt:
        print (p, end = ',')
    print ('])')
    last_day = us_data.index[-1].date() + timedelta(days=1)
    new_last_day = last_day + timedelta(days=extra_rows-1)

    indexes =pd.date_range(last_day.strftime("%m/%d/%y"), new_last_day.strftime("%m/%d/%y")).date

    us_data = us_data.append(pd.DataFrame([[0] * len(us_data.columns)]*extra_rows, index=indexes,columns=us_data.columns))
    #print (us_data)
    x = np.array(range(len(y) + extra_rows))

    #tot['US'] = us_data[country]
    us_data['SIR Fit'] = sir.calc_sir(x, *popt)

    #print (us_data)
    styles = ['o',None]
    linewidths = [0, 1]
    linstiles = [ None, '--']
    fig, ax = plt.subplots()
    ax.set_title(country)

    for col, style, lw, ls in zip(us_data.columns, styles, linewidths, linstiles):
        us_data[col].plot( style=style, lw=lw, ax=ax, linestyle=ls)
    if log_scale:
        ax.set_yscale('log')
    ax.legend()

    plt.figure()
    sir_data = sir.calc_sir(x, *popt, return_sir = True)
    sir_model = pd.DataFrame(sir_data, index = us_data.index, columns = ['S', 'I', 'R'])
    sir_model.iloc[:, 1:].plot()
    plt.figure()
    sir_model.iloc[:, :1].plot()
    return us_data


def main():
    extra_points = 7
    y = create_spread_plot('US', df_infections,extra_points, old_model3)
    
    #plot projection
    nd = y.iloc[1:, :]
    df_der = nd - y.iloc[:-1, :].set_index(nd.index)
    plt.figure()
    df_der['SIR Fit'].plot()
    
    # plot derivative fit
    plt.figure()
    y = y.iloc[:(-1*extra_points), :]
    nd = y.iloc[1:, :]
    df_der = nd - y.iloc[:-1, :].set_index(nd.index)
    df_der.plot()
    #nd.plot()
    plt.show()
if __name__ ==  '__main__':
    main()