"""
forecast_evaluation.py
======================
Volatility forecast evaluation — MSE, QLIKE, MAE, Diebold-Mariano.
Author : Niraj Neupane | github.com/nirajneupane17
"""
import numpy as np
import pandas as pd
from scipy import stats


def mse_loss(actual, forecast):
    m = ~np.isnan(forecast)
    return np.mean((actual[m] - forecast[m])**2)


def qlike_loss(actual, forecast):
    m = (~np.isnan(forecast)) & (forecast > 0) & (actual > 0)
    return np.mean(np.log(forecast[m]) + actual[m]/forecast[m])


def mae_loss(actual, forecast):
    m = ~np.isnan(forecast)
    return np.mean(np.abs(actual[m] - forecast[m]))


def diebold_mariano_test(actual, fc1, fc2, loss='mse'):
    m = ~(np.isnan(fc1) | np.isnan(fc2))
    a, f1, f2 = actual[m], fc1[m], fc2[m]
    e1 = (a-f1)**2 if loss=='mse' else np.abs(a-f1)
    e2 = (a-f2)**2 if loss=='mse' else np.abs(a-f2)
    d = e1 - e2
    dm = d.mean() / (d.std() / np.sqrt(len(d)))
    p  = 2*(1 - stats.norm.cdf(abs(dm)))
    if abs(dm) <= 1.96: conc = 'No significant difference'
    elif dm > 1.96:     conc = 'Model 2 significantly better'
    else:               conc = 'Model 1 significantly better'
    return {'DM_statistic': round(dm,4), 'p_value': round(p,4),
            'significant': abs(dm)>1.96, 'conclusion': conc}


def evaluate_all(actual, forecasts):
    rows = []
    for name, fc in forecasts.items():
        rows.append({'model': name,
                     'MSE':   round(mse_loss(actual,fc), 8),
                     'RMSE':  round(np.sqrt(mse_loss(actual,fc)), 8),
                     'MAE':   round(mae_loss(actual,fc), 8),
                     'QLIKE': round(qlike_loss(actual,fc), 6)})
    return pd.DataFrame(rows).set_index('model').sort_values('QLIKE')
