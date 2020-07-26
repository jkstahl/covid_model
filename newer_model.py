import numpy as np
import seaborn as sns
import scipy, os, pickle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import pandas as pd
from scipy.integrate import odeint
from optparse import OptionParser

old_model = [ 5.35652036e+02,  1.10252107e+00,  2.94664807e-01, -2.51892865e+07]

START_INDEX = 75
ADDITIONAL_ROWS =1



sns.set(rc={'figure.figsize':(15, 7)})
#df = pd.read_csv(r'C:\Users\neoba\Downloads\time_series_covid19_confirmed_global.csv')

df = pd.read_csv(r'C:\Users\neoba\Downloads\us-states.csv')
pop = pd.read_csv(r'C:\Users\neoba\Downloads\State Populations.csv')

pops = {p['State']:p['2018 Population'] for st, p in pop.iterrows()}
print(pops)
df['date'] = pd.to_datetime(df['date'])
#wa = df.loc[df['state'] == 'Washington'].set_index('date')

df = df.iloc[START_INDEX:, :]
df2 = df.copy()
df2 = df2.pivot(index='date', columns='state', values='cases')

df2 = df2[['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Delaware', 'Florida', 'Georgia', 'Oregon', 'Louisiana',  'Nevada', 'North Carolina', 'Oklahoma', 'South Carolina', 'Texas']]
#df2 = df2[['Colorado', 'Idaho', 'Illinois', 'Iowa', 'Kentucky', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Missouri', 'Nebraska', 'New Hampshire', 'New Mexico', 'New Jersey', 'New York', 'North Dakota', 'Pennsylvania', 'Rhode Island', 'Vermont', 'Virginia', 'Wisconsin']]
df2.plot()

df = df2.fillna(0)

class SEIR:
    def __init__(self, N, TI0):
        '''
        TI0 is the total infected including recovered and infected.
        '''
        self.N = N
        self.R0 = TI0 * .1
        self.E0 = TI0 * .1
        self.I0 = max(1, TI0 - (self.R0 + TI0 * .3))
    
    def deriv(self, y, t, k, gamma, delta, x0):
        R_0_start, R_0_end = 3.5,  1
        def logistic_R_0(t):
            return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end
        def beta(t):
            return logistic_R_0(t) * gamma
        S, E, I, R = y
        dSdt = -beta(t) * S * I / self.N
        dEdt = beta(t) * S * I / self.N - delta * E
        dIdt = delta * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt
    
    def fit_model(self, x, y):
        return curve_fit(self.calc_sir, x, y, p0=( 1./10, .1, .1, 1./10), maxfev=10000)

    def calc_sir(self, t,   k, gamma, delta, x0,printit = False):
        '''
        Need to find R0 since we only have infected + recovered
        '''
        S0, E0, I0, R0 = self.N - (self.E0+self.R0+self.I0), self.E0, self.I0, self.R0
        y0 = S0, E0, I0, R0
        ret = odeint(self.deriv, y0, t, args=(  k, gamma, delta, x0))
        ti = ret[:, 2] + ret[:, 3]
        if printit:
            print (ti)
            plt.plot (t, ti)
            plt.show()
        else:
            return ti

class SIR:
    def __init__(self, N, TI0):
        '''
        TI0 is the total infected including recovered and infected.
        '''
        self.N = N
        self.TI0 = TI0
        print (self.N)
        print(self.TI0)
    
    def fit_model(self, x, y):
        return curve_fit(self.calc_sir, x, y, p0=( 1./10, .1, .1, 1./10), maxfev=10000)
    
    def deriv (self, y , t, N, b, gamma, c):
        S, I, R = y
        dSdt = -(b) * S * I / (N - c * t)
        dIdt = (b)* S * I / (N - c * t) - gamma * I
        #dSdt = -(b) * S * I / (N)
        #dIdt = (b)* S * I / (N) - gamma * I
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


#country = 'Texas'
def get_model(df):
    tot = pd.DataFrame(index = df.index)
    models = {}
    x = np.array(range(len(df)))
    for country in df.columns:
        if country not in pops: continue
        print (country)
        y = df[country].astype(float).to_numpy()
        
        sir = SIR(float(pops[country]), y[0])
        popt, pcov = sir.fit_model(x, y)
        print (popt)
        models[country] = popt

    return x, models

def apply_model(df, x, models):
    tot = pd.DataFrame(index = df.index)
    for country in df.columns:
        if country not in pops: continue
        y = df[country].astype(float).to_numpy()
        sir = SIR(float(pops[country]), y[0])
        s = sir.calc_sir(x, *models[country])
        if 'US' not in tot.columns:
            tot['US'] = y
            tot['SIR Fit'] = s
        else:
            tot['US'] += y
            tot['SIR Fit'] += s
    return tot

def main ():
    parser = OptionParser()
    parser.add_option("-f", "--filename", dest="filename", help="Filename to save the model", default=None)

    (options, args) = parser.parse_args()
    
    if options.filename == None or not os.path.exists(options.filename):
        x, models = get_model(df)
        if options.filename != None:
            pickle.dump((x,models), open(options.filename, 'wb'))
    else:
        x, models = pickle.load(open(options.filename, 'rb'))
    x = range(len(df))
    tot = apply_model(df, x, models)
    tot.plot()
    print (tot)
    nd = tot.iloc[1:, :]
    df_der = nd - tot.iloc[:-1, :].set_index(nd.index)
    print (df_der)
    df_der.plot()
    
    plt.show()

if __name__ ==  '__main__':
    main()