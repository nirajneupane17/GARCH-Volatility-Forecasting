"""
har_rv.py
=========
HAR-RV (Heterogeneous Autoregressive Realized Variance).
Corsi (2009) — captures long-memory in volatility via daily,
weekly, and monthly RV components.
Author : Niraj Neupane | github.com/nirajneupane17
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_rv_components(returns, daily=1, weekly=5, monthly=22):
    rv = returns ** 2
    return pd.DataFrame({'RV': rv,
                          'RV_w': rv.rolling(weekly).mean(),
                          'RV_m': rv.rolling(monthly).mean()}).dropna()


def fit_har_rv(returns):
    comps = compute_rv_components(returns)
    y = comps['RV'].shift(-1).dropna()
    X = sm.add_constant(comps[['RV','RV_w','RV_m']].iloc[:-1])
    ols = sm.OLS(y, X).fit()
    return {'model': ols, 'params': ols.params,
            'r_squared': round(ols.rsquared, 4),
            'fitted': ols.fittedvalues, 'residuals': ols.resid}


def har_rv_forecast(har_result, last_rv, last_rv_w, last_rv_m):
    p = har_result['params']
    return max(p['const'] + p['RV']*last_rv + p['RV_w']*last_rv_w + p['RV_m']*last_rv_m, 0)
